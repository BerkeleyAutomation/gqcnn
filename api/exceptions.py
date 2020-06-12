class ApiException(Exception):
    status_code = 404
    message = "Unknown error"

    def __init__(self, message=None, status_code=None, payload=None):
        Exception.__init__(self)
        if message is not None:
            self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


class FileNotFoundException(ApiException):
    status_code = 422
    message = {"file": ["required field"]}

    def __init__(self, message=None, status_code=None, payload=None):
        ApiException.__init__(self)


class FileInputException(ApiException):
    status_code = 422
    message = {"file": ["file input error"]}

    def __init__(self, message=None, status_code=None, payload=None):
        ApiException.__init__(self)