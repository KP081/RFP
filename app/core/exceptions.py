class AppError(Exception):
    status_code = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidFileError(AppError):
    status_code = 400


class RFPNotFoundError(AppError):
    status_code = 404


class RFPNotReadyError(AppError):
    status_code = 409


class LLMError(AppError):
    status_code = 502

class DuplicateRFPError(AppError):
    status_code = 409