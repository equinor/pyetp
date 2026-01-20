import typing


class ETPTransactionFailure(Exception):
    pass


T = typing.TypeVar("T")


def parse_response_errors(
    responses: list[T], expected: typing.Type[T]
) -> list[TypeError]:
    errors = []
    for response in responses:
        if not isinstance(response, expected):
            errors.append(
                TypeError(
                    f"Expected {expected.__name__}, got "
                    f"{response.__class__.__name__} with content: {response}",
                )
            )

    return errors


def raise_response_errors(errors: list[TypeError], location: str) -> None:
    if len(errors) > 0:
        raise ExceptionGroup(
            f"There were {len(errors)} errors in {location}",
            errors,
        )


def parse_and_raise_response_errors(
    responses: list[T], expected: typing.Type[T], location: str
) -> None:
    errors = parse_response_errors(responses, expected)
    raise_response_errors(errors, location)
