#include "queue.h"
#include "logger.h"
#include "runtime_utils.h"

#include <errno.h>
#include <string.h>// For strerror
#include <time.h>

extern Logger *logger;

// Get the timeout time in milliseconds
static struct timespec get_timeout_time( int timeout_ms );

static int max_queue;

Queue *new_queue( int capacity, bool thread_safe ) {
    log_debug( logger, "Creating new queue with capacity %d and thread safety %s",
               capacity, thread_safe ? "enabled" : "disabled" );
    Queue *queue = (Queue *) malloc( sizeof( Queue ) );
    if ( queue == NULL ) {
        return NULL;
    }

    queue->size = 0;
    queue->capacity = capacity;
    queue->thread_safe = thread_safe;
    queue->shutdown = false;
    queue->head = NULL;
    queue->tail = NULL;

    max_queue = 0;

    int ret;

    pthread_mutexattr_t mutex_attr;
    pthread_mutexattr_init( &mutex_attr );
    pthread_mutexattr_settype( &mutex_attr, PTHREAD_MUTEX_ERRORCHECK );

    ret = pthread_mutex_init( &queue->mutex, &mutex_attr );
    pthread_mutexattr_destroy( &mutex_attr );

    if ( ret != 0 ) {
        log_error( logger, "Failed to initialize mutex: %s", strerror( ret ) );
        free( queue );
        return NULL;
    }

    ret = pthread_cond_init( &queue->cond, NULL );
    if ( ret != 0 ) {
        log_error( logger, "Failed to initialize condition variable: %s", strerror( ret ) );
        pthread_mutex_destroy( &queue->mutex );
        free( queue );
        return NULL;
    }

    return queue;
}

int enqueue( Queue *queue, tensors_struct *tensors ) {
    int ret = pthread_mutex_lock( &queue->mutex );
    if ( ret != 0 ) {
        log_error( logger, "Mutex lock failed in enqueue: %s", strerror( ret ) );
        return 1;// Indicate failure
    }

    if ( queue->shutdown ) {
        pthread_mutex_unlock( &queue->mutex );
        return 1;// Indicate failure due to shutdown
    }

    if ( queue->size >= queue->capacity ) {
        // Drop the oldest item (head of the queue)
        QueueItem *old_head = queue->head;
        queue->head = old_head->next;
        queue->size--;

        if ( queue->head == NULL ) {
            // The queue is now empty after removing the head
            queue->tail = NULL;
        }

        // Free the tensors_struct and its data
        if ( old_head->tensors != NULL ) {
            free_tensors_struct( old_head->tensors );
            old_head->tensors = NULL;
        }
        free( old_head );
        old_head = NULL;
    }

    // Proceed to enqueue the new item
    QueueItem *item = (QueueItem *) malloc( sizeof( QueueItem ) );
    if ( item == NULL ) {
        pthread_mutex_unlock( &queue->mutex );
        return 1;// Indicate failure due to malloc error
    }

    item->tensors = tensors;
    item->next = NULL;

    if ( queue->size == 0 ) {
        queue->head = item;
        queue->tail = item;
    } else {
        queue->tail->next = item;
        queue->tail = item;
    }

    queue->size++;

    if ( max_queue < queue->size ) max_queue = queue->size;

    ret = pthread_cond_signal( &queue->cond );
    if ( ret != 0 ) {
        log_warning( logger, "Cond signal failed in enqueue: %s", strerror( ret ) );
    }

    pthread_mutex_unlock( &queue->mutex );

    return 0;// Success
}

tensors_struct *dequeue( Queue *queue, long timeout_ms ) {
    tensors_struct *tensors = NULL;
    int ret = pthread_mutex_lock( &queue->mutex );
    if ( ret != 0 ) {
        log_error( logger, "Mutex lock failed in dequeue: %s", strerror( ret ) );
        return NULL;
    }

    while ( queue->size == 0 && !queue->shutdown ) {
        if ( timeout_ms <= 0 ) {
            ret = pthread_cond_wait( &queue->cond, &queue->mutex );
            if ( ret != 0 ) {
                log_error( logger, "Cond wait failed in dequeue: %s", strerror( ret ) );
                pthread_mutex_unlock( &queue->mutex );
                return NULL;
            }
        } else {
            struct timespec ts = get_timeout_time( timeout_ms );
            ret = pthread_cond_timedwait( &queue->cond, &queue->mutex, &ts );
            if ( ret == ETIMEDOUT ) {
                pthread_mutex_unlock( &queue->mutex );
                return NULL;// Timeout occurred
            } else if ( ret != 0 ) {
                log_error( logger, "Cond timedwait failed in dequeue: %s", strerror( ret ) );
                pthread_mutex_unlock( &queue->mutex );
                return NULL;
            }
        }
    }

    if ( queue->shutdown && queue->size == 0 ) {
        pthread_mutex_unlock( &queue->mutex );
        return NULL;
    }

    if ( queue->size > 0 ) {
        QueueItem *item = queue->head;
        queue->head = item->next;
        queue->size--;

        if ( queue->head == NULL ) {
            queue->tail = NULL;
        }

        tensors = item->tensors;
        free( item );
        item = NULL;
    }

    pthread_mutex_unlock( &queue->mutex );
    return tensors;
}

void shutdown_queue( Queue *queue ) {
    int ret = pthread_mutex_lock( &queue->mutex );
    if ( ret != 0 ) {
        log_error( logger, "Mutex lock failed in shutdown_queue: %s", strerror( ret ) );
        return;
    }

    queue->shutdown = true;

    // Wake up all waiting threads
    ret = pthread_cond_broadcast( &queue->cond );
    if ( ret != 0 ) {
        log_warning( logger, "Cond broadcast failed in shutdown_queue: %s", strerror( ret ) );
    }

    pthread_mutex_unlock( &queue->mutex );
}

void free_queue( Queue *queue ) {
    if ( queue == NULL ) {
        return;
    }

    // Shutdown the queue first
    shutdown_queue( queue );

    // Lock the mutex to safely free the queue
    int ret = pthread_mutex_lock( &queue->mutex );
    if ( ret != 0 ) {
        log_warning( logger, "Mutex lock failed in free_queue: %s", strerror( ret ) );
        // Proceed with freeing resources anyway
    }

    // Free all items in the queue
    QueueItem *current = queue->head;
    while ( current != NULL ) {
        QueueItem *next = current->next;
        if ( current->tensors != NULL ) {
            // Free the tensors_struct and its data
            free_tensors_struct( current->tensors );
            current->tensors = NULL;
        }
        free( current );
        current = next;
    }
    queue->head = NULL;
    queue->tail = NULL;
    queue->size = 0;

    pthread_mutex_unlock( &queue->mutex );

    pthread_mutex_destroy( &queue->mutex );
    pthread_cond_destroy( &queue->cond );

    free( queue );
}

static struct timespec get_timeout_time( int timeout_ms ) {
    struct timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );// Get current time

    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += ( timeout_ms % 1000 ) * 1000000;// Convert ms to ns

    // Normalize timespec (in case nanoseconds overflow)
    while ( ts.tv_nsec >= 1000000000L ) {
        ts.tv_nsec -= 1000000000L;
        ts.tv_sec++;
    }
    return ts;
}
