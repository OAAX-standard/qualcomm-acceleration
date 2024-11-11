#ifndef QUEUE_H
#define QUEUE_H

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "runtime_core.h"

typedef struct QueueItem {
    tensors_struct *tensors;// The item in the queue
    struct QueueItem *next; // Pointer to the next item in the queue
} QueueItem;

typedef struct {
    int size;              // Current number of items in the queue
    int capacity;          // Maximum number of items that the queue can hold
    bool thread_safe;      // Whether the queue should be thread-safe
    bool shutdown;         // Indicates if the queue is shutting down
    QueueItem *head, *tail;// Head and tail of the queue
    pthread_mutex_t mutex; // Mutex for thread-safety
    pthread_cond_t cond;   // Condition variable for thread-safety
} Queue;

/**
 * @brief Create a new queue.
 *
 * @param capacity The maximum number of items that the queue can hold.
 * @param thread_safe Whether the queue should be thread-safe.
 *
 * @return A pointer to the newly created queue.
 */
Queue *new_queue( int capacity, bool thread_safe );

/**
 * @brief Enqueue a new item to the queue.
 *
 * @note The function will only copy the reference to the item, not the item itself. The caller is responsible for managing the memory of the item.
 *
 * @param queue The queue to which the item should be enqueued.
 * @param tensors The item to be enqueued.
 *
 * @return 0 if the item is enqueued successfully, and non-zero otherwise.
 */
int enqueue( Queue *queue, tensors_struct *tensors );

/**
 * @brief Dequeue an item from the queue.
 *
 * @param queue The queue from which the item should be dequeued.
 * @param timeout_ms The timeout in milliseconds. If the timeout is nonpositive, the function will block indefinitely.
 *
 * @return The dequeued item.
 */
tensors_struct *dequeue( Queue *queue, long timeout_ms );

/**
 * @brief Shutdown the queue. No further items can be enqueued.
 *
 * @param queue The queue to be shut down.
 */
void shutdown_queue( Queue *queue );

/**
 * @brief Free the queue and all its items.
 *
 * @warning The function will free the memory of the items in the queue.
 *
 * @param queue The queue to be freed.
 */
void free_queue( Queue *queue );

#endif// QUEUE_H
