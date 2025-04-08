from typing import List
import datetime
import json
import os

class Message:

    def __init__(self):
        self.message: str = ''
        self.data: dict = {}

    def to_dict(self):
        """Convert the message to a dictionary.

        Returns:
            dict: Message as a dictionary
        """
        log = {'Message': self.message}
        if self.data:
            log['Data'] = self.data
        return log

    def __str__(self):
        log = self.to_dict()
        return json.dumps(log, default=str)


class Logs:

    def __init__(self):
        self.messages: List[Message] = []

    def add_message(self, message: str, data: dict = None):
        """Add a new message to the logs.

        Args:
            message (str): Message to add
            data (dict, optional): Data to add. Defaults to None.
        """
        self.messages.append(Message())
        self.messages[-1].message = message
        self.messages[-1].data = data or {}

    def add_data(self, **data):
        """Add data to the last message.

        Note: This may overwrite existing data.

        Args:
            data (dict): Data to add
        """
        try:
            self.messages[-1].data.update(data)
        except:
            self.add_message('<Empty Message>', data)

    def save_as_json(self, path: str = None):
        """Save the logs as a JSON file.

        Args:
            path (str, optional): Path to save the JSON file. Defaults to None.

        Returns:
            str: Path to the saved JSON file
        """
        if path is None:
            path = os.path.join('/', 'tmp', str(datetime.datetime.now()) + '.json')
        with open(path, 'w') as f:
            f.write(str(self))
        return path

    def __str__(self):
        messages_as_dict = [msg.to_dict() for msg in self.messages]
        return json.dumps(messages_as_dict, default=str, indent=2)