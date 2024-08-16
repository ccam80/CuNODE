import threading
import queue
from collections import defaultdict
from multiprocessing import Queue
from time import sleep

class PubSubError(Exception):
    """Custom exception class for PubSub-related errors."""
    pass

class PubSub:
    """
    A thread-safe Publish-Subscribe (Pub-Sub) system that supports both thread-based and
    process-based subscribers. Subscribers can receive messages via callbacks or queues.
    Queue-based subscribers must poll the queue using a "listener" thread.

    Attributes:
        subscribers (defaultdict): A dictionary where keys are event types and
                                   values are lists of either callback functions or queues.
        lock (threading.Lock): A lock to protect the subscribers dictionary from race
                            conditions.
    """

    def __init__(self):
        self.subscribers = defaultdict(list)
        self.lock = threading.Lock()

    def subscribe(self, topic, callback, concurrency=None):
        """
        Registers a subscriber callback function to a specific event type.

        Args:
            topic (str):
                The type of event the subscriber wants to listen to.
            callback (callable):
                The callback function that will be called when the event is published.
                This should take one argument, the published data, of whatever type is
                expected
            concurrency(None (default), 'thread', 'process'):
                If the communicating modules are in the same memory space (threading or no
               concurrency), use a callback function. If the modules are in different processes,
                then set up and return a listener darmon that will call the callback function upon
                receiving data.
        """
        if concurrency == 'process':
            listener = self.subscribe_from_process(topic, callback)
            return listener
        elif concurrency is None or concurrency == 'thread':
            with self.lock:
                self.subscribers[topic].append(callback)

    def subscribe_from_process(self, topic, callback, polling_interval=100):
        """
        Registers a multiprocessing-safe subscriber by creating a queue, and using
        queue.put as the callback fuinction. The user-supplied callback must be
        fed to the listener process. Call BEFORE forking the process.
        Returns the queue

        Args:
            topic (str): The type of event the subscriber wants to listen to.
            callback (callable): Function to call upon event with signature (callback(data))
            polling_interval (int): time in ms between polling for published data.
        Returns:
            listener: A daemon to be started after forking the parent process.

        Gotchas:
            Publishing a None value will cause the listener daemon to close, as
            this is used as an explicit "stop" condition by default. Don't send Nones or
            change this in start_listener.

            Call this function BEFORE forking the process. Call listener.start() AFTER
            forking the process.

        """
        q = Queue()
        listener = listener_daemon(q, callback, polling_interval)

        with self.lock:
            self.subscribers[topic].append(q.put)

        return listener

    def unsubscribe(self, topic, subscriber):
        """
        Unregisters a subscriber by removing their callback or queue from a specific event type.

        Args:
            topic (str): The type of event the subscriber wants to stop listening to.
            subscriber (callable or Queue): The callback function or queue to be removed.

        Raises:
            PubSubError: If the subscriber is not subscribed to the event type.
        """
        with self.lock:
            if subscriber not in self.subscribers[topic]:
                raise PubSubError(f"Subscriber not found for event type '{topic}'.")
            self.subscribers[topic].remove(subscriber)

    def publish(self, topic, data):
        """
        Publishes an event of a specific type with the associated data to all
        subscribers registered for that event type.

        Args:
            topic (str): The type of event being published.
            data (any): The data associated with the event, which will be passed
                        to each subscriber's callback function or put onto their queue.

        Raises:
            PubSubError: If there are no subscribers for the event type.
        """
        with self.lock:
            if topic not in self.subscribers or not self.subscribers[topic]:
                raise PubSubError(f"No subscribers found for event type '{topic}'.")
            for subscriber in self.subscribers[topic]:
                subscriber(data)

    def close_daemons(self, topic):
        """
        Publishes a `None` to a specific event type, signaling subscribers to stop.
        """
        self.publish(topic, None)

    def close_all_daemons(self):
        """
        Sends `None` to all topics to signal their daemons to stop.
        """
        with self.lock:  # Ensure thread-safe access to subscribers
            for topic in self.subscribers.keys():
                self.close_daemons(topic)


class listener_daemon(threading.Thread):
    """
    Listener daemon to poll a queue and run a callback when data is available..

    Args:
        q (Queue): The queue to poll for messages.
        callback (callable): Function to call with signature (callback(data))
        interval_ms (float): The polling interval in seconds (default is 100ms).
    """
    def __init__(self, q, callback, interval_ms=100):

        super().__init__()
        self.q = q
        self.callback = callback
        self.interval = interval_ms/1000
        self.daemon = True

    def run(self):
        while True:
            try:
                data = self.q.get_nowait()
                if data is None:  # Manual thread stop condition
                    break
                self.callback(data)
            except queue.Empty:
                pass
            sleep(self.interval)