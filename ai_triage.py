#!/usr/bin/env python3
"""SQA-Triage Agent.

Clasifica hallazgos de SonarCloud segun ISO/IEC 25010 e ISO/IEC 5055
usando NVIDIA NIM API (compatible con OpenAI).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Final

_LOG: logging.Logger = logging.getLogger("sqa_triage")

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

_ENV_MAP: Final[dict[str, str]] = {
    "sonar_token": "API_SONAR",
    "nvidia_api_key": "API_NVIDIA",
    "sonar_project": "SONAR_PROJECT",
    "nvidia_model": "NVIDIA_MODEL",
    "nvidia_endpoint": "NVIDIA_ENDPOINT",
}

_DEFAULT_ENDPOINT: Final[str] = (
    "https://integrate.api.nvidia.com/v1/chat/completions"
)


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuracion validada del agente."""

    sonar_token: str
    nvidia_api_key: str
    sonar_project: str
    nvidia_model: str = "nvidia/nemotron-3-super-120b-a12b"
    nvidia_endpoint: str = _DEFAULT_ENDPOINT

    @classmethod
    def from_environ(cls) -> "Config":
        """Construye desde variables de entorno. Lanza ValueError si faltan."""
        missing: list[str] = []
        kwargs: dict[str, str] = {}

        for attr, var in _ENV_MAP.items():
            val = os.environ.get(var)
            if val is None or val.strip() == "":
                if attr in ("nvidia_endpoint",):
                    kwargs[attr] = _DEFAULT_ENDPOINT
                    continue
                missing.append(var)
            else:
                kwargs[attr] = val.strip()

        if missing:
            raise ValueError(
                f"Variables de entorno requeridas faltantes: {', '.join(missing)}"
            )
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SonarIssue:
    """Issue normalizado desde la API de SonarCloud."""

    id: str
    title: str
    file: str
    severity: str
    type: str

    @classmethod
    def from_api(cls, raw: dict[str, Any], project_key: str) -> "SonarIssue":
        """Crea una instancia desde un item de la API de SonarCloud."""
        raw_id = raw.get("key", "")
        short_id = raw_id.split("/")[-1][:8] if "/" in raw_id else raw_id[:8]
        return cls(
            id=short_id,
            title=raw.get("message", "Sin titulo"),
            file=raw.get("component", "").replace(f"{project_key}:", ""),
            severity=raw.get("severity", "Unknown"),
            type=raw.get("type", "Unknown"),
        )

    def to_prompt_line(self, idx: int) -> str:
        return (
            f"- ID: {self.id}, Titulo: {self.title}, "
            f"Archivo: {self.file}, Severidad: {self.severity}, "
            f"Tipo: {self.type}"
        )


# ---------------------------------------------------------------------------
# Cliente SonarCloud
# ---------------------------------------------------------------------------

class SonarCloudClient:
    """Cliente minimalista para la API REST de SonarCloud."""

    _BASE: Final[str] = "https://sonarcloud.io/api"

    def __init__(self, token: str, project: str) -> None:
        self._token = token
        self._project = project
        self._auth_header = {"Authorization": f"Bearer {token}"}

    def fetch_open_issues(self, max_results: int = 100) -> list[SonarIssue]:
        """Retorna los issues abiertos del proyecto."""
        url = (
            f"{self._BASE}/issues/search"
            f"?componentKeys={self._project}"
            f"&statuses=OPEN&ps={max_results}"
        )
        req = urllib.request.Request(url, headers=self._auth_header)
        _LOG.info("Consultando SonarCloud: %s issues abiertos", max_results)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload: dict[str, Any] = json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            _LOG.error("Error de conexion SonarCloud: %s", exc)
            raise
        except json.JSONDecodeError as exc:
            _LOG.error("Respuesta invalida de SonarCloud: %s", exc)
            raise

        raw_issues = payload.get("issues", [])
        if not raw_issues:
            _LOG.warning("No se encontraron issues abiertos en SonarCloud. Verificar token y project key. Usando fallback local.")
            return []

        return [
            SonarIssue.from_api(item, self._project) for item in raw_issues
        ]


# ---------------------------------------------------------------------------
# Cliente NVIDIA NIM
# ---------------------------------------------------------------------------

class NvidiaNIMClient:
    """Cliente para NVIDIA NIM API compatible con OpenAI."""

    def __init__(self, api_key: str, model: str, endpoint: str) -> None:
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    _SYSTEM_PROMPT: Final[str] = (
        "Eres SQA-Triage, un agente de aseguramiento de calidad especializado "
        "en clasificar hallazgos de analisis estatico. Recibes una lista de "
        "issues de SonarCloud y debes clasificar cada uno segun:\n\n"
        "1. Caracteristica ISO/IEC 25010 afectada (Adecuacion funcional, "
        "Fiabilidad, Eficiencia de desempeno, Seguridad, Compatibilidad, "
        "Mantenibilidad, Usabilidad, Portabilidad)\n"
        "2. Subcaracteristica especifica (segun ISO/IEC 25010)\n"
        "3. Severidad (Blocker, Critical, Major, Minor, Info)\n"
        "4. Area ISO/IEC 5055 (cuando aplique): Security, Reliability, "
        "Performance Efficiency, Maintainability\n"
        "5. Modulo del codigo afectado\n\n"
        "Reglas estrictas:\n"
        "- No inventes informacion que no este en los datos de entrada\n"
        "- Si un issue no tiene suficiente informacion para clasificarlo, "
        "marcalo como 'Requiere revision humana'\n"
        "- El area ISO 5055 solo se asigna a issues de analisis estatico; "
        "no aplica a defectos funcionales o de documentacion\n"
        "- No sugieras correcciones, solo clasificas\n"
        "- Tu salida es un borrador que sera verificado por un ingeniero "
        "humano antes de usarse\n\n"
        "Formato de salida: tabla markdown con columnas: ID, Caracteristica "
        "ISO 25010, Subcaracteristica, Severidad, Area ISO 5055, Modulo, "
        "Requiere revision humana. Adicionalmente, un resumen con totales "
        "por caracteristica, por severidad, y modulo con mayor concentracion."
    )

    def classify(self, issues: list[SonarIssue]) -> str:
        """Envia los issues al LLM y retorna la clasificacion."""
        user_lines = "\n".join(
            iss.to_prompt_line(i) for i, iss in enumerate(issues, 1)
        )
        user_prompt = (
            f"Ingresa los siguientes issues de SonarCloud para clasificacion:\n"
            f"\n"
            f"ISSUES:\n{user_lines}\n"
            f"\n"
            f"Total de issues: {len(issues)}"
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        _LOG.info("Enviando %d issues a NVIDIA NIM (%s). Esto puede tomar hasta 3 min (cold start)...", len(issues), self._model)
        req = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers,
        )

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data: dict[str, Any] = json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            body = ""
            if hasattr(exc, "read"):
                body = exc.read().decode(errors="replace")[:500]
            _LOG.error("Error NVIDIA NIM [%s]: %s", exc, body)
            raise

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            _LOG.error("Respuesta inesperada de NVIDIA NIM: %s", exc)
            raise ValueError("Estructura de respuesta inesperada") from exc


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

class TriageReport:
    """Genera el reporte markdown final."""

    _TEMPLATE: Final[str] = (
        "# SQA-Triage Report\n"
        "\n"
        "## Ejecucion Automatica\n"
        "\n"
        "- **Modelo:** {model}\n"
        "- **Proyecto:** {project}\n"
        "- **Issues analizados:** {count}\n"
        "- **Fecha:** {date}\n"
        "\n"
        "---\n"
        "\n"
        "{body}\n"
        "\n"
        "---\n"
        "\n"
        "*[IA] Reporte generado automaticamente por SQA-Triage Agent "
        "(NVIDIA NIM).*\n"
        "*Cada clasificacion debe ser verificada por un ingeniero humano "
        "antes de incluirse en el informe.*\n"
    )

    @staticmethod
    def generate(
        model: str,
        project: str,
        issues_count: int,
        body: str,
    ) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return TriageReport._TEMPLATE.format(
            model=model,
            project=project,
            count=issues_count,
            date=date_str,
            body=body.strip(),
        )


# ---------------------------------------------------------------------------
# Fallback: issues de ejemplo (cuando SonarCloud no esta disponible)
# ---------------------------------------------------------------------------

def _fallback_issues() -> list[SonarIssue]:
    """Issues de ejemplo del informe de auditoria."""
    _LOG.warning("Usando datos de fallback (issues del informe)")
    return [
        SonarIssue(
            id="1",
            title="Replace this assertion to not have the same actual and "
                  "expected expression",
            file="tests/test_requests.py",
            severity="Major",
            type="Bug",
        ),
        SonarIssue(
            id="2",
            title="Replace this assertion to not have the same actual and "
                  "expected expression",
            file="tests/test_requests.py",
            severity="Major",
            type="Bug",
        ),
        SonarIssue(
            id="3",
            title="Replace this assertion to not have the same actual and "
                  "expected expression",
            file="tests/test_requests.py",
            severity="Major",
            type="Bug",
        ),
        SonarIssue(
            id="4",
            title='Add the "autouse" marker to this fixture',
            file="tests/test_utils.py",
            severity="Major",
            type="Bug",
        ),
        SonarIssue(
            id="5",
            title='Add a "title" attribute to this iframe',
            file="docs/_templates/sidebar.html",
            severity="Minor",
            type="Bug",
        ),
    ]


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    """Ejecuta el pipeline completo de triage."""
    _setup_logging()

    try:
        config = Config.from_environ()
    except ValueError as exc:
        _LOG.error("Configuracion incompleta: %s", exc)
        sys.exit(1)

    # 1. Obtener issues
    sonar = SonarCloudClient(config.sonar_token, config.sonar_project)
    try:
        issues = sonar.fetch_open_issues()
    except Exception as exc:
        _LOG.warning(
            "No se pudieron obtener issues de SonarCloud (%s). "
            "Usando fallback local.", exc
        )
        issues = _fallback_issues()

    if not issues:
        _LOG.warning("Sin issues para clasificar. Usando fallback.")
        issues = _fallback_issues()

    # 2. Clasificar con NVIDIA
    nim = NvidiaNIMClient(
        config.nvidia_api_key, config.nvidia_model, config.nvidia_endpoint
    )
    try:
        result = nim.classify(issues)
    except Exception as exc:
        _LOG.error("Fallo la clasificacion con NVIDIA NIM: %s", exc)
        sys.exit(2)

    # 3. Generar reporte
    report = TriageReport.generate(
        model=config.nvidia_model,
        project=config.sonar_project,
        issues_count=len(issues),
        body=result,
    )

    output_path = "triage_result.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    _LOG.info("Reporte guardado en %s", output_path)
    print(report)


if __name__ == "__main__":
    main()
