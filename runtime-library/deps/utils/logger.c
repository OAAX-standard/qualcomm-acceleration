#include "logger.h"
#include <sys/time.h>
#include <time.h>

Logger *create_logger( const char *prefix, const char *filename,
                       LogLevel file_level, LogLevel console_level ) {
    Logger *logger = (Logger *) malloc( sizeof( Logger ) );
    if ( logger == NULL ) {
        perror( "Failed to allocate memory for logger" );
        exit( EXIT_FAILURE );
    }

    logger->prefix = strdup( prefix );    // Save the prefix
    logger->filename = strdup( filename );// Save the base filename
    logger->file_index = 0;
    logger->file_level = file_level;
    logger->console_level = console_level;
    pthread_mutex_init( &logger->lock, NULL );

    // Open the initial log file
    char log_filename[1024];
    snprintf( log_filename, sizeof( log_filename ), "%s.%d", logger->filename,
              logger->file_index );
    logger->log_file = fopen( log_filename, "a" );
    if ( logger->log_file == NULL ) {
        perror( "Failed to open log file" );
        free( logger->filename );
        free( logger );
        exit( EXIT_FAILURE );
    }

    return logger;
}

void close_logger( Logger *logger ) {
    if ( logger->log_file != NULL ) {
        fclose( logger->log_file );
    }
    pthread_mutex_destroy( &logger->lock );
    free( logger->filename );
    free( logger->prefix );
    free( logger );
}

void rotate_log_file( Logger *logger ) {
    fclose( logger->log_file );
    logger->file_index++;

    char log_filename[256];
    snprintf( log_filename, sizeof( log_filename ), "%s.%d", logger->filename,
              logger->file_index );
    logger->log_file = fopen( log_filename, "a" );
    if ( logger->log_file == NULL ) {
        perror( "Failed to open new log file" );
        exit( EXIT_FAILURE );
    }
}

void log_message( Logger *logger, LogLevel level, const char *format, ... ) {
    if ( level < logger->file_level && level < logger->console_level ) {
        return;// Skip logging if the level is too low
    }
    pthread_mutex_lock( &logger->lock );

    // Check the size of the log file
    long file_size = ftell( logger->log_file );
    while ( file_size >= MAX_LOG_FILE_SIZE ) {
        rotate_log_file( logger );
        file_size = ftell( logger->log_file );// Update file size
    }
    fseek( logger->log_file, 0, SEEK_END );// Reset file pointer to end

    const char *level_str;
    switch ( level ) {
        case LOG_DEBUG:
            level_str = "DEBUG";
            break;
        case LOG_INFO:
            level_str = "INFO";
            break;
        case LOG_WARNING:
            level_str = "WARNING";
            break;
        case LOG_ERROR:
            level_str = "ERROR";
            break;
        default:
            level_str = "UNKNOWN";
            break;
    }

    struct timeval tv;
    gettimeofday( &tv, NULL );

    time_t now = tv.tv_sec;
    struct tm *localTime = localtime( &now );

    char timeStr[30];// Increased size to accommodate milliseconds
    strftime( timeStr, sizeof( timeStr ), "%Y-%m-%d %H:%M:%S", localTime );

    // Append milliseconds
    char finalTimeStr[40];
    snprintf( finalTimeStr, sizeof( finalTimeStr ), "%s.%03d", timeStr, (int) tv.tv_usec / 1000 );

    va_list args;
    va_start( args, format );

    // build the message as "time - prefix (level): message"
    char message[1024];
    vsnprintf( message, sizeof( message ), format, args );
    if ( level >= logger->file_level && logger->log_file != NULL ) {
        fprintf( logger->log_file, "%s - %s (%s): %s\n", finalTimeStr, logger->prefix,
                 level_str, message );
    }

    // Print to console if the log level is high enough
    if ( level >= logger->console_level ) {
        printf( "%s - %s (%s): %s\n", finalTimeStr, logger->prefix, level_str, message );
    }

    va_end( args );
    pthread_mutex_unlock( &logger->lock );
}
