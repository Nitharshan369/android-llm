#include <jni.h>
#include <string>
#include <vector>
#include <cstring>
#include "llama.h"

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_firstapplication_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* thiz */) {

    const char *model_path =
            "/data/data/com.example.firstapplication/files/smollm2-360m-instruct-q4_k_m.gguf";

    // -------------------------------------------------
    // 1. Init backend
    // -------------------------------------------------
    llama_backend_init();

    // -------------------------------------------------
    // 2. Load model
    // -------------------------------------------------
    llama_model *model = llama_model_load_from_file(
            model_path,
            llama_model_default_params()
    );

    if (!model) {
        llama_backend_free();
        return env->NewStringUTF("Model load failed");
    }

    // -------------------------------------------------
    // 3. Context params
    // -------------------------------------------------
    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx = 1024;
    ctx_params.n_batch = 1;
    ctx_params.n_ubatch = 1;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;

    llama_context *ctx = llama_init_from_model(model, ctx_params);

    if (!ctx) {
        llama_model_free(model);
        llama_backend_free();
        return env->NewStringUTF("Context creation failed");
    }

    // -------------------------------------------------
    // 4. Get vocab
    // -------------------------------------------------
    const llama_vocab *vocab = llama_model_get_vocab(model);

    // -------------------------------------------------
    // 5. Tokenize prompt
    // -------------------------------------------------
    std::string prompt = "Explain python in programming?";

    std::vector<llama_token> tokens(64);

    int n_tokens = llama_tokenize(
            vocab,
            prompt.c_str(),
            (int32_t) prompt.size(),
            tokens.data(),
            (int32_t) tokens.size(),
            true,
            false
    );

    if (n_tokens < 0) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return env->NewStringUTF("Tokenization failed");
    }

    tokens.resize(n_tokens);

    // -------------------------------------------------
    // 6. Create safe batch manually
    // -------------------------------------------------
    llama_batch batch = llama_batch_init(1, 0, 1);

    int decode_result = 0;

    for (int i = 0; i < n_tokens; i++) {
        batch.n_tokens = 1;

        batch.token[0] = tokens[i];
        batch.pos[0] = i;

        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;

        batch.logits[0] = (i == n_tokens - 1);

        decode_result = llama_decode(ctx, batch);

        if (decode_result != 0) {
            break;
        }
    }

    // -------------------------------------------------
    // 7. Decode prompt
    // -------------------------------------------------
    if (decode_result != 0) {
        std::string err =
                "Decode failed. Code = " +
                std::to_string(decode_result);

        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();

        return env->NewStringUTF(err.c_str());
    }

    // -------------------------------------------------
    // 8. Create greedy sampler
    // -------------------------------------------------
    llama_sampler *sampler = llama_sampler_init_greedy();

    if (!sampler) {
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return env->NewStringUTF("Sampler failed");
    }

    // -------------------------------------------------
    // 9. Sample next token and Token to Text
    // -------------------------------------------------
    std::string output = "Generated :";

    for (int step = 0; step < 50; step++) {

        llama_token new_token =
                llama_sampler_sample(sampler, ctx, -1);

        char piece[256];
        memset(piece, 0, sizeof(piece));

        int piece_len = llama_token_to_piece(
                vocab,
                new_token,
                piece,
                sizeof(piece),
                0,
                true
        );

        if (piece_len > 0) {
            output += piece;
        }

        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        // feed generated token back
        batch.n_tokens = 1;
        batch.token[0] = new_token;
        batch.pos[0] = n_tokens + step;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        if (llama_decode(ctx, batch) != 0) {
            break;
        }
    }

    // -------------------------------------------------
    // 11. Cleanup
    // -------------------------------------------------
    llama_sampler_free(sampler);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    // -------------------------------------------------
    // 12. Return result
    // -------------------------------------------------
    return env->NewStringUTF(output.c_str());
}