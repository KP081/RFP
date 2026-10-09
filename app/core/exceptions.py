class APPError(Exception):
    status_code = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidFileError(APPError):
    status_code = 400


class RFPNotFoundError(APPError):
    status_code = 404


class RFPNotReadyError(APPError):
    status_code = 409


class LLMError(APPError):
    status_code = 502
