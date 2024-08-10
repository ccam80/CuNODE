# -*- coding: utf-8 -*-
"""
Created on Wed August 10 19:30:00 2024

@author: Chris Cameron
"""
import threading
from collections import defaultdict
import logging

class pubsub:
    """
    A simple thread-safe Publish-Subscribe (Pub-Sub) system for communication
    between GUI widgets and submodules. Allows multiple subscribers to
    listen to multiple event types published by different publishers.

    Attributes:
        subscribers (defaultdict): A dictionary where keys are event types
                                   and values are lists of subscriber callback
                                   functions. Defaultdict allows an indexing by
                                   a nonexistent key to create a new blank list.
        lock (threading.Lock): A lock to protect the subscribers dict from race
                                conditions.
    """

    def __init__(self):
        """
        Initializes the PubSub system with an empty dict of subscribers and a
        threading lock to ensure thread safety.
        """
        self.subscribers = defaultdict(list)
        self.lock = threading.Lock()

    def subscribe(self, topic, subscriber_function):
        """
        Registers a subscriber callback function to a specific event type.

        Args:
            topic (str): The type of event the subscriber wants to listen to.
            subscriber_function (callable): The callback function that will be called
                                            when the event is published.
        """
        with self.lock:
            self.subscribers[topic].append(subscriber_function)

    def unsubscribe(self, event_type, subscriber):
        """
        Unregisters a subscriber callback function from a specific event type.

        Args:
            event_type (str): The type of event the subscriber wants to stop listening to.
            subscriber (callable): The callback function to be removed from the subscribers list.
        """
        with self.lock:
            if subscriber in self.subscribers[event_type]:
                self.subscribers[event_type].remove(subscriber)
            else:
                logging.debug(f"Unsubscribe called but {subscriber} isn't registered to remove")

    def publish(self, event_type, data):
        """
        Publishes an event of a specific type with the associated data to all
        subscribers registered for that event type.

        Args:
            event_type (str): The type of event being published.
            data (any): The data associated with the event, which will be passed
                        to each subscriber's callback function.
        """
        with self.lock:
            if event_type in self.subscribers:
                if self.subscribers[event_type] == []:
                    logging.debug(f"No one is listening to {event_type}")
                for subscriber in self.subscribers[event_type]:
                    subscriber(data)

            else:
                logging.debug(f"publish was called for a non-registered tpoic {event_type}")