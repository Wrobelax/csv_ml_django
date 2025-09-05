from django import template

register = template.Library()

@register.filter
def format_model(value) -> str:
    if not value:
        return value
    return str(value).replace("_", " ").title()