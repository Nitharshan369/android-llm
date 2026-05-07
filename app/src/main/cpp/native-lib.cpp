#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>

#include "llama.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "LLAMA", __VA_ARGS__)

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_firstapplication_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* thiz */) {

    const char *model_path =
            "/data/data/com.example.firstapplication/files/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf";

    llama_backend_init();

    llama_model_params model_params =
            llama_model_default_params();

    model_params.n_gpu_layers = 0;

    llama_model *model =
            llama_model_load_from_file(
                    model_path,
                    model_params
            );

    if (!model) {
        llama_backend_free();
        return env->NewStringUTF("Model load failed");
    }

    llama_context_params ctx_params =
            llama_context_default_params();

    ctx_params.n_ctx = 256;

    ctx_params.n_batch = 64;
    ctx_params.n_ubatch = 64;

    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;

    llama_context *ctx =
            llama_init_from_model(
                    model,
                    ctx_params
            );

    if (!ctx) {

        llama_model_free(model);
        llama_backend_free();

        return env->NewStringUTF(
                "Context creation failed"
        );
    }

    const llama_vocab *vocab =
            llama_model_get_vocab(model);

    std::string prompt =
            "<|system|>\n"
            "You are a helpful assistant.</s>\n"
            "<|user|>\n"
            "Explain Python simply.</s>\n"
            "<|assistant|>\n";

    std::vector<llama_token> tokens(256);

    int n_tokens = llama_tokenize(
            vocab,
            prompt.c_str(),
            prompt.length(),
            tokens.data(),
            tokens.size(),
            true,
            true
    );

    if (n_tokens < 0) {

        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();

        return env->NewStringUTF(
                "Tokenization failed"
        );
    }

    tokens.resize(n_tokens);

    LOGI("Token count = %d", n_tokens);

    llama_batch batch =
            llama_batch_init(
                    n_tokens,
                    0,
                    1
            );

    for (int i = 0; i < n_tokens; i++) {

        batch.token[i] = tokens[i];
        batch.pos[i] = i;

        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;

        batch.logits[i] =
                (i == n_tokens - 1) ? 1 : 0;
    }

    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {

        llama_batch_free(batch);

        llama_free(ctx);

        llama_model_free(model);

        llama_backend_free();

        return env->NewStringUTF(
                "Prompt decode failed"
        );
    }

    llama_sampler *sampler =
            llama_sampler_chain_init(
                    llama_sampler_chain_default_params()
            );

    llama_sampler_chain_add(
            sampler,
            llama_sampler_init_temp(0.8f)
    );

    llama_sampler_chain_add(
            sampler,
            llama_sampler_init_top_k(40)
    );

    llama_sampler_chain_add(
            sampler,
            llama_sampler_init_top_p(
                    0.95f,
                    1
            )
    );

    llama_sampler_chain_add(
            sampler,
            llama_sampler_init_dist(1234)
    );

    std::string output = "Generated:\n\n";

    for (int step = 0; step < 100; step++) {

        llama_token token =
                llama_sampler_sample(
                        sampler,
                        ctx,
                        -1
                );

        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        char piece[256];

        int len =
                llama_token_to_piece(
                        vocab,
                        token,
                        piece,
                        sizeof(piece),
                        0,
                        true
                );

        if (len > 0) {
            output += std::string(piece, len);
        }

        batch.n_tokens = 1;

        batch.token[0] = token;

        batch.pos[0] =
                n_tokens + step;

        batch.n_seq_id[0] = 1;

        batch.seq_id[0][0] = 0;

        batch.logits[0] = 1;

        if (llama_decode(ctx, batch) != 0) {
            break;
        }
    }

    llama_sampler_free(sampler);

    llama_batch_free(batch);

    llama_free(ctx);

    llama_model_free(model);

    llama_backend_free();

    return env->NewStringUTF(output.c_str());
}