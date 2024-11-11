#include "runtime_core.h"
#include "logger.h"
#include "queue.h"
#include "runtime_utils.h"
#include "threads.h"
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>// For strerror
#include <time.h>

#ifndef ONNXRUNTIME_API_VERSION
#define ONNXRUNTIME_API_VERSION 15
#endif

#define N_QNN_OPTIONS 5

#define QUEUE_CAPACITY 100// The maximum number of items that a queue can hold
#define THREAD_SAFE true

#define RUNTIME_ORT_CORE_EXEC( return_code, error ) ( {int32_t code = return_code; if (code != 0) {error = code; goto cleanup;} } )

#define INIT_LOGGER                                                                   \
    if ( logger == NULL ) {                                                           \
        logger = create_logger( runtime_name(), "runtime.log", log_level, LOG_INFO ); \
        if ( logger == NULL ) {                                                       \
            printf( "Error: RUNTIME - Failed to create logger\n" );                   \
            return 1;                                                                 \
        }                                                                             \
    }
#define READ_ENV_INT( name, var )             \
    {                                         \
        const char *env = getenv( name );     \
        if ( env != NULL ) var = atoi( env ); \
    }

static int n_duplicates = 1;
static int n_threads_per_duplicate = 4;
static LogLevel log_level = LOG_INFO;

const OrtApi *api;

OrtSession **sessions = NULL;
OrtRunOptions **run_options = NULL;
OrtAllocator **allocators = NULL;
OrtMemoryInfo **memory_infos = NULL;
OrtEnv **envs = NULL;
OrtSessionOptions **session_options = NULL;

// Queues
static Queue *input_queue = NULL, *output_queue = NULL;

// Logger
Logger *logger = NULL;

// Running thread
int *session_ids = NULL;
pthread_t *threads = NULL;
atomic_int stop_thread = 0;

// Prototype functions
static void *inference_loop( void *arg );
static int runtime_inference_execution( int session_id, tensors_struct *input_tensors, tensors_struct *output_tensors );

int runtime_initialization_with_args( int length, const char **keys, const void **values ) {
    // Initialize the logger
    READ_ENV_INT( "RUNTIME_LOG_LEVEL", log_level );
    INIT_LOGGER

    log_info( logger, "Initializing ONNX Runtime with arguments" );
    for ( int i = 0; i < length; i++ ) {
        if ( strcmp( keys[i], "n_duplicates" ) == 0 ) {
            n_duplicates = *(int *) values[i];
        } else if ( strcmp( keys[i], "n_threads_per_duplicate" ) == 0 ) {
            n_threads_per_duplicate = *(int *) values[i];
        } else {
            log_warning( logger, "Unknown key '%s'", keys[i] );
        }
    }
    return runtime_initialization();
}

int runtime_initialization() {
    // Initialize the logger
    READ_ENV_INT( "RUNTIME_LOG_LEVEL", log_level );
    INIT_LOGGER

    log_info( logger, "Initializing OnnxRuntime" );

    // get the correct api version
    api = OrtGetApiBase()->GetApi( ONNXRUNTIME_API_VERSION );

    // Allocate memory for the objects
    session_ids = (int *) malloc( sizeof( int ) * n_duplicates );
    run_options = (OrtRunOptions **) malloc( sizeof( OrtRunOptions * ) * n_duplicates );
    allocators = (OrtAllocator **) malloc( sizeof( OrtAllocator * ) * n_duplicates );
    memory_infos = (OrtMemoryInfo **) malloc( sizeof( OrtMemoryInfo * ) * n_duplicates );
    envs = (OrtEnv **) malloc( sizeof( OrtEnv * ) * n_duplicates );
    session_options = (OrtSessionOptions **) malloc( sizeof( OrtSessionOptions * ) * n_duplicates );
    sessions = (OrtSession **) malloc( sizeof( OrtSession * ) * n_duplicates );
    threads = (pthread_t *) malloc( sizeof( pthread_t ) * n_duplicates );

    if ( !session_ids || !run_options || !allocators || !memory_infos || !envs || !session_options || !sessions || !threads ) {
        log_error( logger, "Memory allocation failed during initialization" );
        return 1;
    }

    // Get the logging level
    char *logging_level = getenv( "ORT_LOGGING_LEVEL" );
    if ( logging_level == NULL ) {
        log_error( logger, "ORT_LOGGING_LEVEL environment variable is not set." );
        exit( 1 );
    }
    int logging_level_int = atoi( logging_level );

    // Get the backend path
    char *backend_path = getenv( "QNN_BACKEND_PATH" );
    if ( backend_path == NULL ) {
        log_error( logger, "QNN_BACKEND_PATH environment variable is not set" );
        exit( 1 );
    }

    for ( int i = 0; i < n_duplicates; i++ ) {
        // create the environment
        if ( runtime_core_process_status( api->CreateEnv( logging_level_int, "ORT_LOGGER", &envs[i] ) ) != 0 ) {
            log_error( logger, "Failed to create ORT environment" );
            return 1;
        }

        // Create session options
        if ( runtime_core_process_status( api->CreateSessionOptions( &session_options[i] ) ) != 0 ) {
            log_error( logger, "Failed to create session options" );
            return 1;
        }

        // set session options
        log_info( logger, "QNN_BACKEND_PATH: %s", backend_path );
        const char *const provider_options_keys[N_QNN_OPTIONS] = {
                "backend_path", "htp_graph_finalization_optimization_mode",
                "disable_cpu_ep_fallback", "htp_performance_mode", "profiling_level" };
        const char *const provider_options_values[N_QNN_OPTIONS] = {
                backend_path, "1", "1", "high_performance", "detailed" };

        runtime_core_process_status( api->SetSessionGraphOptimizationLevel( session_options[i], ORT_ENABLE_BASIC ) );
        runtime_core_process_status( api->SetIntraOpNumThreads( session_options[i], n_threads_per_duplicate ) );
        runtime_core_process_status( api->SessionOptionsAppendExecutionProvider(
                session_options[i], "QNN", provider_options_keys, provider_options_values, N_QNN_OPTIONS ) );
        runtime_core_process_status( api->SetInterOpNumThreads( session_options[i], 1 ) );
        runtime_core_process_status( api->SetSessionExecutionMode( session_options[i], ORT_SEQUENTIAL ) );

        char **providers = NULL;
        int number_providers = 0;
        runtime_core_process_status( api->GetAvailableProviders( &providers, &number_providers ) );
        for ( int j = 0; j < number_providers; ++j )
            log_info( logger, "Provider id: %i - name: %s", j, providers[j] );
        runtime_core_process_status( api->ReleaseAvailableProviders( providers, number_providers ) );

        // create run options
        if ( runtime_core_process_status( api->CreateRunOptions( &run_options[i] ) ) != 0 ) {
            log_error( logger, "Failed to create run options" );
            return 1;
        }
    }

    // Initialize the queues
    input_queue = new_queue( QUEUE_CAPACITY, THREAD_SAFE );
    output_queue = new_queue( QUEUE_CAPACITY, THREAD_SAFE );

    if ( !input_queue || !output_queue ) {
        log_error( logger, "Failed to create queues" );
        return 1;
    }

    log_debug( logger, "n_duplicates: %d - n_threads_per_duplicate: %d - log_level: %d",
               n_duplicates, n_threads_per_duplicate, log_level );

    return 0;
}

int runtime_model_loading( const char *file_path ) {
    log_info( logger, "Reading ONNX file from '%s' ...", file_path );

    // Create a session
    for ( int i = 0; i < n_duplicates; i++ ) {
        session_ids[i] = i;
        if ( runtime_core_process_status( api->CreateSession( envs[i], file_path, session_options[i], &sessions[i] ) ) != 0 ) {
            log_error( logger, "Failed to create session" );
            return 1;
        }

        // create allocator
        if ( runtime_core_process_status( api->CreateCpuMemoryInfo( OrtArenaAllocator, OrtMemTypeDefault, &memory_infos[i] ) ) != 0 ) {
            log_error( logger, "Failed to create memory info" );
            return 1;
        }

        if ( runtime_core_process_status( api->CreateAllocator( sessions[i], memory_infos[i], &allocators[i] ) ) != 0 ) {
            log_error( logger, "Failed to create allocator" );
            return 1;
        }
    }

    stop_thread = 0;

    // Start the inference loops
    for ( int i = 0; i < n_duplicates; i++ ) {
        if ( pthread_create( &threads[i], NULL, inference_loop, &session_ids[i] ) != 0 ) {
            log_error( logger, "Failed to create inference thread" );
            return 1;
        }
    }

    return 0;
}

static void *inference_loop( void *arg ) {
    int session_id = *( (int *) arg );
    log_info( logger, "Starting the inference thread for session %d.", session_id );
    int exit_code = 0;
    while ( 1 ) {
        tensors_struct *input_tensors = dequeue( input_queue, 200 );// timeout after 200ms to check if the thread should stop
        if ( input_tensors != NULL ) {

            log_debug( logger, "Running inference for session %d.", session_id );
            tensors_struct *output_tensors = (tensors_struct *) malloc( sizeof( tensors_struct ) );
            if ( output_tensors == NULL ) {
                log_warning( logger, "Memory allocation failed for output_tensors" );
                free_tensors_struct( input_tensors );
                continue;
            }
            memset( output_tensors, 0, sizeof( tensors_struct ) );// Initialize to zero

            exit_code = runtime_inference_execution( session_id, input_tensors, output_tensors );
            if ( exit_code != 0 ) {
                log_warning( logger, "Inference execution failed with code %d", exit_code );
                free_tensors_struct( output_tensors );
                output_tensors = NULL;
            }

            if ( output_tensors != NULL && exit_code == 0 ) {
                log_debug( logger, "Enqueueing output tensors." );
                if ( enqueue( output_queue, output_tensors ) != 0 ) {
                    log_warning( logger, "Failed to enqueue output tensors" );
                    free_tensors_struct( output_tensors );
                }
            } else {
                log_warning( logger, "No output tensors to enqueue." );
            }

            free_tensors_struct( input_tensors );
        }
        // Check if the thread should stop
        if ( atomic_load( &stop_thread ) ) {
            log_info( logger, "Stopping inference thread: %i", session_id );
            break;
        }
    }
    return NULL;
}

int send_input( tensors_struct *input_tensors ) {
    log_debug( logger, "Sending input tensors to the queue." );
    if ( enqueue( input_queue, input_tensors ) != 0 ) {
        log_error( logger, "Failed to enqueue input tensors" );
        return 1;
    }
    return 0;
}

int receive_output( tensors_struct **output_tensors ) {
    log_debug( logger, "Receiving output tensors from the queue." );
    tensors_struct *output = dequeue( output_queue, 200 );
    if ( output == NULL ) {
        log_debug( logger, "There's no output tensors available." );
        *output_tensors = NULL;
        return 1;
    }

    *output_tensors = output;
    return 0;
}

static int runtime_inference_execution( int session_id, tensors_struct *input_tensors, tensors_struct *output_tensors ) {
    int error = 0;
    int number_inputs = (int) input_tensors->num_tensors;
    uint8_t **inputs = (uint8_t **) input_tensors->data;
    int64_t **input_shapes = (int64_t **) input_tensors->shapes;
    size_t *input_ranks = (size_t *) input_tensors->ranks;
    size_t *input_sizes = NULL;
    char **input_names = NULL;
    int *input_dtypes = NULL;
    char **output_names = NULL;
    int *output_dtypes = NULL;
    OrtValue **input_values = NULL;
    OrtValue **output_values = NULL;
    void **outputs = NULL;
    int number_outputs = 0;
    size_t **output_shapes = NULL;
    size_t *output_ranks = NULL;

    input_sizes = (size_t *) malloc( sizeof( size_t ) * number_inputs );
    if ( input_sizes == NULL ) {
        log_error( logger, "Memory allocation failed for input_sizes" );
        error = 1;
        goto cleanup;
    }

    for ( int i = 0; i < number_inputs; i++ ) {
        input_sizes[i] = 1;
        for ( size_t j = 0; j < input_ranks[i]; j++ )
            input_sizes[i] *= input_shapes[i][j];
    }

    log_debug( logger, "Reading input names from ONNX file" );
    input_names = runtime_core_get_input_names( sessions[session_id], allocators[session_id], &number_inputs, &input_dtypes );
    if ( input_names == NULL || input_dtypes == NULL ) {
        log_error( logger, "Failed to get input names or data types" );
        error = 1;
        goto cleanup;
    }

    // Make sure that the number of input tensors is the same as the one passed to the runtime
    if ( number_inputs != (int) input_tensors->num_tensors ) {
        log_error( logger, "The number of input tensors does not match the number of input tensors in the model" );
        log_error( logger, "Expected: %d, Got: %zu", number_inputs, input_tensors->num_tensors );
        error = 1;
        goto cleanup;
    }

    log_debug( logger, "Reading output names from ONNX file" );
    output_names = runtime_core_get_output_names( sessions[session_id], allocators[session_id], &number_outputs, &output_dtypes );
    if ( output_names == NULL || output_dtypes == NULL ) {
        log_error( logger, "Failed to get output names or data types" );
        error = 1;
        goto cleanup;
    }

    input_values = (OrtValue **) malloc( sizeof( OrtValue * ) * number_inputs );
    if ( input_values == NULL ) {
        log_error( logger, "Memory allocation failed for input_values" );
        error = 1;
        goto cleanup;
    }
    memset( input_values, 0, sizeof( OrtValue * ) * number_inputs );

    output_values = (OrtValue **) malloc( sizeof( OrtValue * ) * number_outputs );
    if ( output_values == NULL ) {
        log_error( logger, "Memory allocation failed for output_values" );
        error = 1;
        goto cleanup;
    }
    memset( output_values, 0, sizeof( OrtValue * ) * number_outputs );

    for ( int i = 0; i < number_inputs; ++i ) {
        RUNTIME_ORT_CORE_EXEC( runtime_core_process_status(
                                       api->CreateTensorWithDataAsOrtValue( memory_infos[session_id],
                                                                            inputs[i],
                                                                            input_sizes[i] * runtime_util_get_sizeof_onnx_type( input_dtypes[i] ),
                                                                            input_shapes[i],
                                                                            input_ranks[i],
                                                                            input_dtypes[i],
                                                                            &input_values[i] ) ),
                               error );
    }

    log_debug( logger, "Inference run started" );
    RUNTIME_ORT_CORE_EXEC( runtime_core_process_status( api->Run( sessions[session_id],
                                                                  run_options[session_id],
                                                                  (const char *const *) input_names,
                                                                  (const OrtValue *const *) input_values,
                                                                  number_inputs,
                                                                  (const char *const *) output_names,
                                                                  number_outputs,
                                                                  output_values ) ),
                           error );

    log_debug( logger, "Inference run completed" );
    // Clean up input values
    for ( int i = 0; i < number_inputs; i++ ) {
        if ( input_names[i] ) free( input_names[i] );
        if ( input_values[i] ) api->ReleaseValue( input_values[i] );
    }
    free( input_names );
    free( input_dtypes );
    free( input_sizes );
    free( input_values );
    input_names = NULL;
    input_dtypes = NULL;
    input_sizes = NULL;
    input_values = NULL;

    outputs = (void **) malloc( sizeof( void * ) * number_outputs );
    if ( outputs == NULL ) {
        log_error( logger, "Memory allocation failed for outputs" );
        error = 1;
        goto cleanup;
    }
    memset( outputs, 0, sizeof( void * ) * number_outputs );

    output_shapes = (size_t **) malloc( sizeof( size_t * ) * number_outputs );
    if ( output_shapes == NULL ) {
        log_error( logger, "Memory allocation failed for output_shapes" );
        error = 1;
        goto cleanup;
    }
    memset( output_shapes, 0, sizeof( size_t * ) * number_outputs );

    output_ranks = (size_t *) malloc( sizeof( size_t ) * number_outputs );
    if ( output_ranks == NULL ) {
        log_error( logger, "Memory allocation failed for output_ranks" );
        error = 1;
        goto cleanup;
    }

    for ( int i = 0; i < number_outputs; ++i ) {
        OrtTensorTypeAndShapeInfo *type_shape = NULL;

        // get shape information
        RUNTIME_ORT_CORE_EXEC( runtime_core_process_status( api->GetTensorTypeAndShape( output_values[i], &type_shape ) ), error );

        // get output size
        size_t size = 0;
        RUNTIME_ORT_CORE_EXEC( runtime_core_process_status( api->GetTensorShapeElementCount( type_shape, &size ) ), error );

        // get output rank
        size_t rank = 0;
        RUNTIME_ORT_CORE_EXEC( runtime_core_process_status( api->GetDimensionsCount( type_shape, &rank ) ), error );
        output_ranks[i] = rank;

        // get output shape
        output_shapes[i] = (size_t *) malloc( sizeof( int64_t ) * rank );
        if ( output_shapes[i] == NULL ) {
            log_error( logger, "Memory allocation failed for output_shapes[%d]", i );
            api->ReleaseTensorTypeAndShapeInfo( type_shape );
            error = 1;
            goto cleanup;
        }
        RUNTIME_ORT_CORE_EXEC( runtime_core_process_status( api->GetDimensions( type_shape, (int64_t *) output_shapes[i], rank ) ), error );

        // get output value
        size_t output_bytes = size * runtime_util_get_sizeof_onnx_type( output_dtypes[i] );
        outputs[i] = malloc( output_bytes );
        if ( outputs[i] == NULL ) {
            log_error( logger, "Memory allocation failed for outputs[%d]", i );
            api->ReleaseTensorTypeAndShapeInfo( type_shape );
            error = 1;
            goto cleanup;
        }
        void *tmp = NULL;
        RUNTIME_ORT_CORE_EXEC( runtime_core_process_status( api->GetTensorMutableData( output_values[i], &tmp ) ), error );
        memcpy( outputs[i], tmp, output_bytes );

        api->ReleaseTensorTypeAndShapeInfo( type_shape );
        api->ReleaseValue( output_values[i] );
    }
    free( output_values );
    output_values = NULL;

    // Assign the output tensors to the output_tensors pointer
    output_tensors->num_tensors = number_outputs;
    output_tensors->ranks = output_ranks;
    output_tensors->shapes = output_shapes;
    output_tensors->data_types = (tensor_data_type *) output_dtypes;
    output_tensors->data = outputs;
    output_tensors->names = output_names;

    // Set pointers to NULL to avoid double free in cleanup
    output_ranks = NULL;
    output_shapes = NULL;
    outputs = NULL;
    output_dtypes = NULL;
    output_names = NULL;

cleanup:
    // Clean up in case of errors
    if ( input_names ) {
        for ( int i = 0; i < number_inputs; i++ ) {
            if ( input_names[i] ) free( input_names[i] );
        }
        free( input_names );
    }
    if ( input_dtypes ) free( input_dtypes );
    if ( input_sizes ) free( input_sizes );
    if ( input_values ) {
        for ( int i = 0; i < number_inputs; i++ ) {
            if ( input_values[i] ) api->ReleaseValue( input_values[i] );
        }
        free( input_values );
    }
    if ( output_names ) {
        for ( int i = 0; i < number_outputs; i++ ) {
            if ( output_names[i] ) free( output_names[i] );
        }
        free( output_names );
    }
    if ( output_dtypes ) free( output_dtypes );
    if ( output_values ) {
        for ( int i = 0; i < number_outputs; i++ ) {
            if ( output_values[i] ) {
                api->ReleaseValue( output_values[i] );
            }
        }
        free( output_values );
    }
    if ( output_shapes ) {
        for ( int i = 0; i < number_outputs; i++ ) {
            if ( output_shapes[i] ) free( output_shapes[i] );
        }
        free( output_shapes );
    }
    if ( output_ranks ) free( output_ranks );
    if ( outputs ) {
        for ( int i = 0; i < number_outputs; i++ ) {
            if ( outputs[i] ) free( outputs[i] );
        }
        free( outputs );
    }
    return error;
}

int runtime_destruction() {
    log_info( logger, "Releasing all objects created by the runtime" );
    // Stop the thread
    atomic_store( &stop_thread, 1 );

    // Wait for all threads to finish
    for ( int i = 0; i < n_duplicates; i++ ) {
        pthread_join( threads[i], NULL );
    }

    // Clean up the queues
    free_queue( input_queue );
    free_queue( output_queue );

    // Clean up the session
    for ( int i = 0; i < n_duplicates; i++ ) {
        api->ReleaseRunOptions( run_options[i] );
        api->ReleaseSession( sessions[i] );
        api->ReleaseMemoryInfo( memory_infos[i] );
        api->ReleaseAllocator( allocators[i] );
        api->ReleaseEnv( envs[i] );
        api->ReleaseSessionOptions( session_options[i] );
    }
    free( threads );
    free( allocators );
    free( memory_infos );
    free( envs );
    free( sessions );
    free( run_options );
    free( session_ids );
    free( session_options );

    // destroy logger
    if ( logger != NULL ) {
        close_logger( logger );
        logger = NULL;
    }

    return 0;
}

const char *runtime_error_message() {
    return "Check the stdout and log files for the error message.";
}

const char *runtime_version() {
    return "0.1.0";
}

const char *runtime_name() {
    return "OnnxRuntime";
}
