"""Strategy prompt templates.

This module provides template loading functionality for strategy prompts.
Templates are stored as .txt files in this directory.

Built-in templates:
- default: Conservative, risk-aware trading
- aggressive: Momentum/breakout focused trading
- insane: Maximum aggression (testing only)
- funding_rate: Funding rate arbitrage for perpetuals
"""

from pathlib import Path
from typing import Dict, List, Optional

# Template directory
TEMPLATES_DIR = Path(__file__).parent

# Built-in template registry
BUILTIN_TEMPLATES = {
    "default": "default.txt",
    "aggressive": "aggressive.txt",
    "insane": "insane.txt",
    "nof1": "nof1.txt",
}


class TemplateNotFoundError(Exception):
    """Raised when a template cannot be found."""
    pass


class TemplateLoader:
    """Load strategy prompt templates from files.

    Supports:
    - Built-in templates by name (e.g., "default", "aggressive")
    - Custom templates by file path
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize template loader.

        Args:
            templates_dir: Custom templates directory. Defaults to built-in templates.
        """
        self._templates_dir = templates_dir or TEMPLATES_DIR
        self._cache: Dict[str, str] = {}

    def list_templates(self) -> List[str]:
        """List all available template names."""
        templates = list(BUILTIN_TEMPLATES.keys())

        # Add any custom .txt files in directory
        if self._templates_dir.exists():
            for f in self._templates_dir.glob("*.txt"):
                name = f.stem
                if name not in templates:
                    templates.append(name)

        return sorted(templates)

    def load(self, template_id: str) -> str:
        """Load a template by ID or name.

        Args:
            template_id: Template name (e.g., "default") or file path

        Returns:
            Template content as string

        Raises:
            TemplateNotFoundError: If template cannot be found
        """
        # Check cache first
        if template_id in self._cache:
            return self._cache[template_id]

        content = self._load_template(template_id)
        self._cache[template_id] = content
        return content

    def _load_template(self, template_id: str) -> str:
        """Internal template loading logic."""
        # Try built-in template by name
        if template_id in BUILTIN_TEMPLATES:
            filename = BUILTIN_TEMPLATES[template_id]
            path = self._templates_dir / filename
            if path.exists():
                return path.read_text(encoding="utf-8")

        # Try as direct filename in templates dir
        path = self._templates_dir / f"{template_id}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")

        # Try as absolute or relative path
        path = Path(template_id)
        if path.exists():
            return path.read_text(encoding="utf-8")

        # Template not found
        available = ", ".join(self.list_templates())
        raise TemplateNotFoundError(
            f"Template '{template_id}' not found. "
            f"Available templates: {available}"
        )

    def get_template_path(self, template_id: str) -> Optional[Path]:
        """Get the file path for a template.

        Args:
            template_id: Template name or path

        Returns:
            Path to template file, or None if not found
        """
        # Built-in template
        if template_id in BUILTIN_TEMPLATES:
            path = self._templates_dir / BUILTIN_TEMPLATES[template_id]
            if path.exists():
                return path

        # Direct filename
        path = self._templates_dir / f"{template_id}.txt"
        if path.exists():
            return path

        # Absolute/relative path
        path = Path(template_id)
        if path.exists():
            return path

        return None

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()


# Global template loader instance
_loader: Optional[TemplateLoader] = None


def get_template_loader() -> TemplateLoader:
    """Get or create the global template loader."""
    global _loader
    if _loader is None:
        _loader = TemplateLoader()
    return _loader


def load_template(template_id: str) -> str:
    """Convenience function to load a template.

    Args:
        template_id: Template name or path

    Returns:
        Template content
    """
    return get_template_loader().load(template_id)


def list_templates() -> List[str]:
    """Convenience function to list available templates."""
    return get_template_loader().list_templates()
