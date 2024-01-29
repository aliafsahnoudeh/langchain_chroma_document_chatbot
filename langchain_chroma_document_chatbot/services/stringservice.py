def snake_to_camel(snake_string: str) -> str:
    parts = snake_string.split("_")
    return parts[0].lower() + "".join(part.capitalize() for part in parts[1:])
