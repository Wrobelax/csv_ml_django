from django import template

register = template.Library()

@register.filter
def format_model(value: str) -> str:
    return value.replace("_", " ").title()