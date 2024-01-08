#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>

int main(int argc, char **argv)
{
    // parse separator argument
    std::string separator = "";
    std::string arg = "";
    for (int i = 1; i < argc; i++)
    {
        arg = argv[i];
        if (arg == "--separator")
        {
            separator = argv[i + 1];

            // remove separator from argv to avoid errors later
            argv[i] = argv[argc - 2];
            argv[i + 1] = argv[argc - 1];
            argc -= 2;

            break;
        }
    }

    // parse llamacpp arguments
    gpt_params params;
    llama_sampling_params &sparams = params.sparams;
    if (gpt_params_parse(argc, argv, params) == false)
    {
        fprintf(stderr, "%s: error: failed to parse command line arguments\n", __func__);
        return 1;
    }

    // prompt contains the prompt_file contents by default
    // split prompt into multiple prompts using separator
    std::vector<std::string> prompts;
    size_t pos = 0;
    while ((pos = params.prompt.find(separator)) != std::string::npos)
    {
        std::string prompt = params.prompt.substr(0, pos);
        prompts.push_back(prompt);
        params.prompt.erase(0, pos + separator.length());
    }

    // initialize the llama backend
    llama_backend_init(params.numa);

    // load the model
    llama_model_params model_params = llama_model_params_from_gpt_params(params);
    llama_model *model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    LOG_TEE("%s\n", get_system_info(params).c_str());

    // total length of the sequences including the prompt
    int n_len = params.n_predict;

    for (auto &prompt : prompts)
    {
        params.prompt = prompt;
        llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
        llama_context *ctx = llama_new_context_with_model(model, ctx_params);

        std::vector<llama_token> tokens_list = llama_tokenize(ctx, params.prompt, true);

        const int n_ctx = llama_n_ctx(ctx);
        const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

        // make sure the KV cache is big enough to hold all the prompt and generated tokens
        if (n_kv_req > n_ctx)
        {
            LOG_TEE("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
            LOG_TEE("%s:        either reduce n_len or increase n_ctx\n", __func__);
            return 1;
        }

        // print the prompt token-by-token
        fprintf(stderr, "\n");

        for (auto id : tokens_list)
        {
            fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
        }

        fflush(stderr);

        // create a llama_batch
        // we use this object to submit token data for decoding
        llama_batch batch = llama_batch_init(n_ctx, 0, 1);

        // evaluate the initial prompt
        for (size_t i = 0; i < tokens_list.size(); i++)
        {
            llama_batch_add(batch, tokens_list[i], i, {0}, false);
        }

        // llama_decode will output logits only for the last token of the prompt
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx, batch) != 0)
        {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
            return 1;
        }

        struct llama_sampling_context *ctx_sampling = llama_sampling_init(sparams);

        int n_cur = batch.n_tokens;
        int n_decode = 0;

        const auto t_main_start = ggml_time_us();

        while (n_cur <= n_len)
        {
            // sample the next token
            {
                const llama_token id = llama_sampling_sample(ctx_sampling, ctx, NULL, batch.n_tokens - 1);

                llama_sampling_accept(ctx_sampling, ctx, id, true);

                // is it an end of stream?
                if (id == llama_token_eos(model) || n_cur == n_len)
                {
                    std::cout << std::endl;

                    break;
                }

                std::cout << llama_token_to_piece(ctx, id).c_str();
                std::cout << std::flush;

                // prepare the next batch
                llama_batch_clear(batch);

                // push this new token for next evaluation
                llama_batch_add(batch, id, n_cur, {0}, true);

                n_decode += 1;
            }

            n_cur += 1;

            // evaluate the current batch with the transformer model
            if (llama_decode(ctx, batch))
            {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                return 1;
            }
        }

        LOG_TEE("\n");

        const auto t_main_end = ggml_time_us();

        LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
                __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

        llama_print_timings(ctx);

        fprintf(stderr, "\n");

        llama_batch_free(batch);

        llama_sampling_free(ctx_sampling);

        llama_kv_cache_clear(ctx);

        llama_free(ctx);
    }

    llama_free_model(model);

    llama_backend_free();

    return 0;
}
