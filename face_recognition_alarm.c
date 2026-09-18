#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "onnxruntime_c_api.h"
#include <microhttpd.h>
#include <limits.h>
#include <curl/curl.h>

#define DEFAULT_PORT 6000
#define DEFAULT_FACE_IMAGE_PATH "face_detect/detected_face_latest.jpg"
#define DEFAULT_MODEL_PATH "finetuned_model.onnx"
#define DEFAULT_REF_EMBEDDING_PATH "ref_embedding.txt"
#define DEFAULT_SIMILARITY_THRESHOLD 0.35f

static const char* getenv_or_default(const char* name, const char* def) {
    const char* val = getenv(name);
    return (val && val[0] != '\0') ? val : def;
}

float* load_reference_embedding(const char* filename, size_t* out_size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open reference embedding file: %s\n", filename);
        return NULL;
    }
    size_t emb_size = 512;
    float* embedding = (float*)malloc(emb_size * sizeof(float));
    if (!embedding) {
        fclose(fp);
        return NULL;
    }
    for (size_t i = 0; i < emb_size; i++) {
        fscanf(fp, "%f", &embedding[i]);
    }
    fclose(fp);
    *out_size = emb_size;
    return embedding;
}


float cosine_similarity(const float* a, const float* b, size_t size) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < size; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-6);
}

/*
  Preprocess an image:
    This function loads the image, resizes it to 112x112,
    normalizes using mean=[0.5,0.5,0.5] and std=[0.5,0.5,0.5],
    and outputs a float array of size 1x3x112x112.
    This implementation uses stb_image and stb_image_resize.
*/
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

float* preprocess_image(const char* image_path) {
    int width, height, channels;
    // Load image as 8-bit, forcing 3 channels (RGB)
    unsigned char* img_data = stbi_load(image_path, &width, &height, &channels, 3);
    if (!img_data) {
        fprintf(stderr, "Error: Unable to load image %s\n", image_path);
        return NULL;
    }

    int target_width = 112, target_height = 112;
    unsigned char* resized_data = (unsigned char*)malloc(target_width * target_height * 3 * sizeof(unsigned char));
    if (!resized_data) {
        fprintf(stderr, "Error: Unable to allocate memory for resized image\n");
        stbi_image_free(img_data);
        return NULL;
    }


    if (width != target_width || height != target_height) {
        if (!stbir_resize_uint8(img_data, width, height, 0,
                                resized_data, target_width, target_height, 0, 3)) {
            fprintf(stderr, "Error: Failed to resize image\n");
            free(resized_data);
            stbi_image_free(img_data);
            return NULL;
        }
    } else {
        memcpy(resized_data, img_data, target_width * target_height * 3 * sizeof(unsigned char));
    }
    stbi_image_free(img_data);

    // Allocate buffer for normalized float data (HWC order)
    int num_pixels = target_width * target_height * 3;
    float* float_img = (float*)malloc(num_pixels * sizeof(float));
    if (!float_img) {
        fprintf(stderr, "Error: Unable to allocate memory for float image\n");
        free(resized_data);
        return NULL;
    }

    // Normalize: map [0,255] to [-1,1] using: ((pixel/255.0)-0.5)/0.5
    for (int i = 0; i < num_pixels; i++) {
        float_img[i] = (((float)resized_data[i]) / 255.0f - 0.5f) / 0.5f;
    }
    free(resized_data);

    // Convert from HWC to CHW order
    int output_size = 1 * 3 * target_width * target_height;
    float* output = (float*)malloc(output_size * sizeof(float));
    if (!output) {
        fprintf(stderr, "Error: Unable to allocate memory for output tensor\n");
        free(float_img);
        return NULL;
    }

    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < target_height; h++) {
            for (int w = 0; w < target_width; w++) {
                int index_hwc = h * target_width * 3 + w * 3 + c;
                int index_chw = c * target_width * target_height + h * target_width + w;
                output[index_chw] = float_img[index_hwc];
            }
        }
    }
    free(float_img);
    return output;
}

void send_telegram_alert(const char *image_path, const char *caption) {
    const char* bot_token = getenv("TELEGRAM_BOT_TOKEN");
    const char* chat_id = getenv("TELEGRAM_CHAT_ID");
    if (!bot_token || !chat_id || bot_token[0] == '\0' || chat_id[0] == '\0') {
        fprintf(stderr, "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; skipping alert.\n");
        return;
    }

    CURL *curl;
    CURLcode res;

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (curl) {
        char url[512];
        snprintf(url, sizeof(url), "https://api.telegram.org/bot%s/sendPhoto", bot_token);


        curl_mime *mime;
        curl_mimepart *part;
        mime = curl_mime_init(curl);

        // Add chat_id field.
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "chat_id");
        curl_mime_data(part, chat_id, CURL_ZERO_TERMINATED);

        // Attach image file.
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "photo");
        curl_mime_filedata(part, image_path);

        // Add caption field.
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "caption");
        curl_mime_data(part, caption, CURL_ZERO_TERMINATED);

        // Set CURL options.
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);

        // Perform the request.
        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            fprintf(stderr, "Failed to send Telegram alert: %s\n", curl_easy_strerror(res));
        } else {
            printf("Telegram alert sent successfully!\n");
        }

        curl_mime_free(mime);
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
}

// HTTP request handler: when /trigger is hit, process the saved image.
static enum MHD_Result request_handler(void *cls, struct MHD_Connection *connection,
                           const char *url, const char *method,
                           const char *version, const char *upload_data,
                           size_t *upload_data_size, void **con_cls) {
    if (strcmp(url, "/trigger") != 0) {
        return MHD_NO;
    }

    printf("Trigger received from Python!\n");

    const char* face_image_path = getenv_or_default("FACE_IMAGE_PATH", DEFAULT_FACE_IMAGE_PATH);
    printf("Using detected face image: %s\n", face_image_path);

    // Preprocess the image: returns a float array of size 1x3x112x112.
    float* input_tensor_values = preprocess_image(face_image_path);
    if (!input_tensor_values) {
        return MHD_NO;
    }
    size_t input_tensor_size = 1 * 3 * 112 * 112;
    int64_t input_shape[] = {1, 3, 112, 112};

    // Initialize ONNX Runtime.
    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env = NULL;
    OrtSessionOptions* session_options = NULL;
    OrtSession* session = NULL;
    OrtMemoryInfo* memory_info = NULL;
    OrtValue* input_tensor = NULL;
    OrtValue* output_tensor = NULL;
    float* output_data = NULL;
    OrtStatus* status = NULL;

    // Create environment.
    status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "face_recognition", &env);
    if (status != NULL) {
        fprintf(stderr, "Error creating environment: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        free(input_tensor_values);
        return MHD_NO;
    }

    // Create session options.
    status = ort->CreateSessionOptions(&session_options);
    if (status != NULL) {
        fprintf(stderr, "Error creating session options: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        free(input_tensor_values);
        ort->ReleaseEnv(env);
        return MHD_NO;
    }

    // Set number of intra op threads.
    status = ort->SetIntraOpNumThreads(session_options, 1);
    if (status != NULL) {
        fprintf(stderr, "Error setting intra op threads: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        free(input_tensor_values);
        ort->ReleaseSessionOptions(session_options);
        ort->ReleaseEnv(env);
        return MHD_NO;
    }

    // Path to the ONNX model.
    const char* model_path = getenv_or_default("MODEL_PATH", DEFAULT_MODEL_PATH);
    status = ort->CreateSession(env, model_path, session_options, &session);
    if (status != NULL) {
        fprintf(stderr, "Error creating session: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        free(input_tensor_values);
        ort->ReleaseSessionOptions(session_options);
        ort->ReleaseEnv(env);
        return MHD_NO;
    }

    // Create input tensor.
    status = ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status != NULL) {
        fprintf(stderr, "Error creating CPU memory info: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        free(input_tensor_values);
        ort->ReleaseSession(session);
        ort->ReleaseSessionOptions(session_options);
        ort->ReleaseEnv(env);
        return MHD_NO;
    }

    status = ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values,
                                                  input_tensor_size * sizeof(float),
                                                  input_shape, 4,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                  &input_tensor);
    if (status != NULL) {
        fprintf(stderr, "Error creating tensor: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        ort->ReleaseMemoryInfo(memory_info);
        free(input_tensor_values);
        ort->ReleaseSession(session);
        ort->ReleaseSessionOptions(session_options);
        ort->ReleaseEnv(env);
        return MHD_NO;
    }
    ort->ReleaseMemoryInfo(memory_info);


    const char* input_names[] = {"input"};
    const char* output_names[] = {"embedding"};

    // inference.
    status = ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor,
                      1, output_names, 1, &output_tensor);
    if (status != NULL) {
        fprintf(stderr, "Error during inference: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        ort->ReleaseValue(input_tensor);
        free(input_tensor_values);
        ort->ReleaseSession(session);
        ort->ReleaseSessionOptions(session_options);
        ort->ReleaseEnv(env);
        return MHD_NO;
    }

    // Get output data.
    status = ort->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status != NULL) {
        fprintf(stderr, "Error getting output tensor data: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        ort->ReleaseValue(input_tensor);
        ort->ReleaseValue(output_tensor);
        free(input_tensor_values);
        ort->ReleaseSession(session);
        ort->ReleaseSessionOptions(session_options);
        ort->ReleaseEnv(env);
        return MHD_NO;
    }

    size_t ref_size;
    const char* ref_embedding_path = getenv_or_default("REF_EMBEDDING_PATH", DEFAULT_REF_EMBEDDING_PATH);
    float* ref_embedding = load_reference_embedding(ref_embedding_path, &ref_size);
    if (ref_embedding == NULL) {
        printf("Failed to load reference embedding.\n");
    } else {
        float similarity = cosine_similarity(ref_embedding, output_data, ref_size);
        printf("Cosine similarity: %f\n", similarity);
        float threshold = (float)atof(getenv_or_default("SIMILARITY_THRESHOLD", "0.35"));
        if (similarity < threshold) {
            printf("Alarm: Unknown face detected!\n");
            send_telegram_alert(face_image_path, "UNKNOWN PERSON");
        } else {
            printf("Face recognized as yours.\n");
        }
        free(ref_embedding);
    }

    // Clean up ONNX Runtime objects.
    ort->ReleaseValue(output_tensor);
    ort->ReleaseValue(input_tensor);
    ort->ReleaseSession(session);
    ort->ReleaseSessionOptions(session_options);
    ort->ReleaseEnv(env);
    free(input_tensor_values);

    const char *response_str = "Trigger processed.";
    struct MHD_Response *response = MHD_create_response_from_buffer(strlen(response_str),
                                                    (void *)response_str,
                                                    MHD_RESPMEM_PERSISTENT);
    int ret = MHD_queue_response(connection, MHD_HTTP_OK, response);
    MHD_destroy_response(response);
    return ret;
}

int main(void) {
    int port = atoi(getenv_or_default("HTTP_PORT", "6000"));
    if (port <= 0) port = DEFAULT_PORT;

    struct MHD_Daemon *daemon;
    daemon = MHD_start_daemon(MHD_USE_SELECT_INTERNALLY, port, NULL, NULL,
                              &request_handler, NULL, MHD_OPTION_END);
    if (NULL == daemon) {
        fprintf(stderr, "Failed to start HTTP server.\n");
        return 1;
    }
    printf("C HTTP server running on port %d...\n", port);
    // The server runs until a key is pressed.
    getchar();
    MHD_stop_daemon(daemon);
    return 0;
}
