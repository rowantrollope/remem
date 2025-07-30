#!/usr/bin/env python3
"""
Debug utilities for memory system with nice ANSI formatting.
"""

import os
import sys
from typing import Any, Optional


class Colors:
    """ANSI color codes for terminal output."""
    # Basic colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # Bright colors
    BRIGHT_RED = '\033[91;1m'
    BRIGHT_GREEN = '\033[92;1m'
    BRIGHT_YELLOW = '\033[93;1m'
    BRIGHT_BLUE = '\033[94;1m'
    BRIGHT_MAGENTA = '\033[95;1m'
    BRIGHT_CYAN = '\033[96;1m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    # Background colors
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled."""
    return os.getenv("MEMORY_DEBUG", "false").lower() == "true"


def is_verbose_enabled() -> bool:
    """Check if verbose mode is enabled."""
    return os.getenv("MEMORY_VERBOSE", "false").lower() == "true"


def supports_color() -> bool:
    """Check if terminal supports color output."""
    return (
        hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and
        os.getenv('TERM') != 'dumb' and
        os.getenv('NO_COLOR') is None
    )


def colorize(text: str, color: str) -> str:
    """Add color to text if terminal supports it."""
    if supports_color():
        return f"{color}{text}{Colors.RESET}"
    return text


def debug_print(message: str, prefix: str = "DEBUG", color: str = Colors.GRAY) -> None:
    """Print debug message only if debug mode is enabled."""
    if is_debug_enabled():
        formatted_prefix = colorize(f"[{prefix}]", color)
        print(f"{formatted_prefix} {message}")


def verbose_print(message: str, prefix: str = "INFO", color: str = Colors.BLUE) -> None:
    """Print verbose message if verbose or debug mode is enabled."""
    if is_verbose_enabled() or is_debug_enabled():
        formatted_prefix = colorize(f"[{prefix}]", color)
        print(f"{formatted_prefix} {message}")


def success_print(message: str) -> None:
    """Print success message with green color."""
    icon = colorize("✅", Colors.GREEN)
    print(f"{icon} {message}")


def error_print(message: str) -> None:
    """Print error message with red color."""
    icon = colorize("❌", Colors.RED)
    print(f"{icon} {message}")


def warning_print(message: str) -> None:
    """Print warning message with yellow color."""
    icon = colorize("⚠️", Colors.YELLOW)
    print(f"{icon} {message}")


def info_print(message: str) -> None:
    """Print info message with blue color."""
    icon = colorize("ℹ️", Colors.BLUE)
    print(f"{icon} {message}")


def memory_search_print(vectorstore: str, query: str, top_k: int, min_similarity: float) -> None:
    """Print formatted memory search information."""
    if is_debug_enabled():
        search_icon = colorize("🔍", Colors.CYAN)
        vectorstore_colored = colorize(vectorstore, Colors.BRIGHT_BLUE)
        query_colored = colorize(f'"{query}"', Colors.WHITE)
        params = colorize(f"(top_k: {top_k}, min_similarity: {min_similarity})", Colors.GRAY)
        print(f"{search_icon} {vectorstore_colored}: Searching memories: {query_colored} {params}")


def memory_result_print(included_count: int, excluded_count: int, min_similarity: float) -> None:
    """Print formatted memory filtering results."""
    if is_debug_enabled():
        filter_icon = colorize("🔍", Colors.CYAN)
        total = included_count + excluded_count
        result = colorize(f"{total} → {included_count}", Colors.WHITE)
        threshold = colorize(f"min_similarity: {min_similarity}", Colors.GRAY)
        print(f"{filter_icon} Similarity filtering result: {result} memories ({threshold})")


def memory_extraction_print(extracted_count: int, context: str = "") -> None:
    """Print formatted memory extraction results."""
    if is_debug_enabled():
        if extracted_count > 0:
            brain_icon = colorize("🧠", Colors.MAGENTA)
            count_colored = colorize(str(extracted_count), Colors.BRIGHT_GREEN)
            print(f"{brain_icon} MEMORY: Auto-extracted {count_colored} NEW memories from {context}")
        else:
            search_icon = colorize("🔍", Colors.CYAN)
            print(f"{search_icon} MEMORY: No new memories extracted - information already captured or not valuable")


def section_header(title: str, width: int = 50) -> None:
    """Print a formatted section header."""
    separator = "=" * width
    title_colored = colorize(title, Colors.BRIGHT_CYAN)
    separator_colored = colorize(separator, Colors.GRAY)
    print(f"\n{separator_colored}")
    print(f"{title_colored}")
    print(f"{separator_colored}")


def format_memory_item(memory: dict, index: int) -> str:
    """Format a memory item for display."""
    score = memory.get('score', memory.get('relevance_score', 0))
    text = memory.get('text', memory.get('final_text', ''))
    
    # Truncate text if too long
    if len(text) > 60:
        text = text[:57] + "..."
    
    index_colored = colorize(f"#{index}", Colors.GRAY)
    score_colored = colorize(f"{score:.3f}", Colors.YELLOW)
    text_colored = colorize(f"'{text}'", Colors.WHITE)
    
    return f"   {index_colored}: Score: {score_colored} - {text_colored}"


def format_user_response(response: str) -> str:
    """Format the final user response with clear separation."""
    if not response.strip():
        return ""
    
    # Add a clear separator before the response
    separator = colorize("─" * 60, Colors.GRAY)
    response_label = colorize("Agent Response:", Colors.BRIGHT_GREEN)
    
    return f"\n{separator}\n{response_label}\n{response}\n{separator}"


def clear_line() -> None:
    """Clear the current line in terminal."""
    if supports_color():
        print('\r\033[K', end='')


def progress_indicator(message: str) -> None:
    """Show a progress indicator."""
    if is_verbose_enabled() or is_debug_enabled():
        spinner = colorize("⏳", Colors.YELLOW)
        print(f"{spinner} {message}...", end='', flush=True)


def format_grounding_display(storage_result: dict) -> str:
    """Format grounding information with enhanced visual display.

    Shows the original text with color-coded replacements and a clear
    summary of what was changed during contextual grounding.

    Args:
        storage_result: Dictionary containing grounding information

    Returns:
        Formatted string with color-coded grounding display
    """
    if not storage_result.get('grounding_applied'):
        return ""

    original_text = storage_result.get('original_text', '')
    final_text = storage_result.get('final_text', '')
    grounding_info = storage_result.get('grounding_info', {})
    changes_made = grounding_info.get('changes_made', [])

    if not changes_made:
        return ""

    # Create the enhanced display
    lines = []
    lines.append(colorize("🌍 Contextual Grounding Applied:", Colors.BRIGHT_CYAN))
    lines.append("")

    # Show original text with highlighted replacements
    display_text = original_text

    # Sort changes by position (if available) or by length (longest first to avoid overlap issues)
    sorted_changes = sorted(changes_made, key=lambda x: len(x.get('original', '')), reverse=True)

    # Apply color coding to show what was replaced
    for change in sorted_changes:
        original_word = change.get('original', '')

        if original_word in display_text:
            # Color the original word in red (what was replaced)
            colored_original = colorize(original_word, Colors.BG_RED + Colors.WHITE)
            display_text = display_text.replace(original_word, colored_original, 1)  # Replace only first occurrence

    lines.append(f"   {colorize('Original:', Colors.GRAY)} {display_text}")
    lines.append(f"   {colorize('Result:', Colors.GRAY)}   {colorize(final_text, Colors.BRIGHT_GREEN)}")
    lines.append("")

    # Show detailed changes
    lines.append(f"   {colorize('Changes Made:', Colors.BRIGHT_YELLOW)}")
    for i, change in enumerate(changes_made, 1):
        original_word = change.get('original', '')
        replacement = change.get('replacement', '')
        change_type = change.get('type', 'unknown')

        # Format the change with colors
        original_colored = colorize(f'"{original_word}"', Colors.BRIGHT_RED)
        replacement_colored = colorize(f'"{replacement}"', Colors.BRIGHT_GREEN)
        type_colored = colorize(f'({change_type})', Colors.CYAN)
        arrow = colorize('→', Colors.GRAY)

        lines.append(f"   {colorize(f'{i}.', Colors.GRAY)} {original_colored} {arrow} {replacement_colored} {type_colored}")

    return "\n".join(lines)


def progress_done() -> None:
    """Clear progress indicator."""
    if is_verbose_enabled() or is_debug_enabled():
        clear_line()
