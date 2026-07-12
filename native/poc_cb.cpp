// PoC: verify that ggml_backend_sched eval callbacks can (a) capture the
// residual stream (l_out-<il> tensors) and (b) modify it mid-graph such that
// the change propagates to the final logits.
//
// Usage: ./poc_cb <model.gguf>

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct cb_state {
    // capture
    std::map<int, std::vector<float>> captured; // layer -> [n_tokens * d]
    int64_t d_model = 0;
    int64_t n_tokens = 0;
    // intervention
    bool do_intervene = false;
    int target_layer = -1;
    float add_value = 0.0f;
};

static int parse_l_out(const char * name) {
    // accept "l_out-12" and split-prefixed variants like "CPU#l_out-12#0"
    const char * p = strstr(name, "l_out-");
    if (!p) return -1;
    return atoi(p + 6);
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    cb_state * st = (cb_state *) user_data;
    const int il = parse_l_out(t->name);
    if (ask) {
        return il >= 0;
    }
    if (il < 0) return true;

    const int64_t d = t->ne[0], n = t->ne[1];
    st->d_model = d; st->n_tokens = n;

    std::vector<float> buf(d * n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, d * n * sizeof(float));
    } else {
        fprintf(stderr, "unexpected type %s for %s\n", ggml_type_name(t->type), t->name);
        return true;
    }

    if (st->do_intervene && il == st->target_layer) {
        for (auto & v : buf) v += st->add_value;
        ggml_backend_tensor_set(t, buf.data(), 0, d * n * sizeof(float));
        printf("  [intervene] added %.1f to l_out-%d (%lldx%lld)\n", st->add_value, il, (long long)d, (long long)n);
    }

    st->captured[il] = std::move(buf);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 1; }
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
    }, nullptr);

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    cb_state st;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = cparams.n_ubatch = 256;
    cparams.cb_eval = cb_eval;
    cparams.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    const char * text = "Once upon a time there was a";
    std::vector<llama_token> toks(64);
    int n = llama_tokenize(vocab, text, strlen(text), toks.data(), toks.size(), true, true);
    toks.resize(n);
    printf("tokenized %d tokens\n", n);

    auto run = [&](bool intervene, float add) -> std::vector<float> {
        llama_memory_clear(llama_get_memory(ctx), true);
        st.captured.clear();
        st.do_intervene = intervene;
        st.target_layer = 2;
        st.add_value = add;
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; i++) {
            batch.token[i] = toks[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1; // all positions: keeps last-layer l_out full-width
        }
        batch.n_tokens = n;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
        llama_batch_free(batch);
        const float * logits = llama_get_logits_ith(ctx, n - 1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        return std::vector<float>(logits, logits + n_vocab);
    };

    // baseline
    auto logits0 = run(false, 0.0f);
    printf("captured %zu layers, d_model=%lld, n_tokens=%lld\n",
           st.captured.size(), (long long)st.d_model, (long long)st.n_tokens);
    for (auto & kv : st.captured) {
        double ss = 0; for (float v : kv.second) ss += (double)v * v;
        printf("  l_out-%d  n=%zu  rms=%.4f\n", kv.first, kv.second.size() / (size_t)st.d_model,
               sqrt(ss / kv.second.size()));
    }
    // determinism check
    auto logits0b = run(false, 0.0f);
    double max_dd = 0;
    for (size_t i = 0; i < logits0.size(); i++) max_dd = std::max(max_dd, (double)fabs(logits0[i] - logits0b[i]));
    printf("determinism: max |dlogit| across identical runs = %.3e\n", max_dd);

    // intervened
    auto logits1 = run(true, 10.0f);
    double max_d = 0; int argmax0 = 0, argmax1 = 0;
    for (size_t i = 0; i < logits0.size(); i++) {
        max_d = std::max(max_d, (double)fabs(logits0[i] - logits1[i]));
        if (logits0[i] > logits0[argmax0]) argmax0 = i;
        if (logits1[i] > logits1[argmax1]) argmax1 = i;
    }
    printf("intervention: max |dlogit| = %.4f  argmax %d -> %d\n", max_d, argmax0, argmax1);
    printf(max_d > 1e-3 ? "POC OK: modification propagated to logits\n"
                        : "POC FAIL: logits unchanged\n");

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
