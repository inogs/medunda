"""This module provides utility functions for argument parsing in Medunda tools."""

from datetime import datetime


def date_from_str(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")
