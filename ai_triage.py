#!/usr/bin/env python3
"""
SQA-Triage Agent
Clasifica issues de SonarCloud segun ISO/IEC 25010 e ISO/IEC 5055
usando NVIDIA NIM API (compatible con OpenAI).
"""
import json
import os
import sys
import urllib.request

SONAR_TOKEN = os.environ.get('API_SONAR', os.environ.get('SONAR_TOKEN', ''))
NVIDIA_API_KEY = os.environ.get('API_NVIDIA', os.environ.get('NVIDIA_API_KEY', ''))
PROJECT_KEY = os.environ.get('SONAR_PROJECT', 'daniell-estrada_requests')
NVIDIA_MODEL = os.environ.get('NVIDIA_MODEL', 'meta/llama-3.1-70b-instruct')
NVIDIA_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'

SYSTEM_PROMPT = """Eres SQA-Triage, un agente de aseguramiento de calidad especializado en clasificar hallazgos de analisis estatico. Recibes una lista de issues de SonarCloud y debes clasificar cada uno segun:

1. Caracteristica ISO/IEC 25010 afectada (Adecuacion funcional, Fiabilidad, Eficiencia de desempeno, Seguridad, Compatibilidad, Mantenibilidad, Usabilidad, Portabilidad)
2. Subcaracteristica especifica (segun ISO/IEC 25010)
3. Severidad (Blocker, Critical, Major, Minor, Info)
4. Area ISO/IEC 5055 (cuando aplique): Security, Reliability, Performance Efficiency, Maintainability
5. Modulo del codigo afectado

Reglas estrictas:
- No inventes informacion que no este en los datos de entrada
- Si un issue no tiene suficiente informacion para clasificarlo, marcalo como "Requiere revision humana"
- El area ISO 5055 solo se asigna a issues de analisis estatico del codigo fuente; no aplica a defectos funcionales o de documentacion
- No sugieras correcciones, solo clasificas
- Tu salida es un borrador que sera verificado por un ingeniero humano antes de usarse

Formato de salida: tabla markdown con columnas: ID, Caracteristica ISO 25010, Subcaracteristica, Severidad, Area ISO 5055, Modulo, Requiere revision humana. Adicionalmente, un resumen con totales por caracteristica, por severidad, y modulo con mayor concentracion."""


def get_sonar_issues():
    """Obtiene issues abiertos de SonarCloud vía API REST."""
    url = f'https://sonarcloud.io/api/issues/search?componentKeys={PROJECT_KEY}&statuses=OPEN&ps=100'
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {SONAR_TOKEN}'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"ERROR obteniendo issues de SonarCloud: {e}", file=sys.stderr)
        return []
    
    issues = []
    for issue in data.get('issues', []):
        issues.append({
            'id': issue['key'].split('/')[-1][:8],
            'title': issue['message'],
            'file': issue['component'].replace(PROJECT_KEY + ':', ''),
            'severity': issue['severity'],
            'type': issue['type']
        })
    return issues


def get_modules(issues):
    """Obtiene modulos unicos de los issues."""
    modules = {}
    for issue in issues:
        f = issue['file']
        if f not in modules:
            modules[f] = 0
        modules[f] += 1
    return modules


def call_nvidia(issues, modules):
    """Llama a NVIDIA NIM API para clasificar issues."""
    user_prompt = f"Ingresa los siguientes issues de SonarCloud para clasificacion:\n\n"
    user_prompt += "ISSUES:\n"
    for issue in issues:
        user_prompt += f"- ID: {issue['id']}, Titulo: {issue['title']}, Archivo: {issue['file']}, Severidad: {issue['severity']}, Tipo: {issue['type']}\n"
    
    user_prompt += "\nMODULOS:\n"
    for module, count in modules.items():
        user_prompt += f"- {module}: {count} issue(s)\n"

    payload = {
        'model': NVIDIA_MODEL,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt}
        ],
        'temperature': 0.1,
        'max_tokens': 2048
    }
    
    req = urllib.request.Request(
        NVIDIA_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {NVIDIA_API_KEY}',
            'Content-Type': 'application/json'
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data['choices'][0]['message']['content']
    except Exception as e:
        print(f"ERROR llamando a NVIDIA API: {e}", file=sys.stderr)
        if hasattr(e, 'read'):
            err_body = e.read().decode()
            print(f"Respuesta del servidor: {err_body[:500]}", file=sys.stderr)
        return "ERROR: No se pudo obtener clasificacion del agente SQA-Triage."


def main():
    print("=" * 60)
    print("SQA-Triage Agent - Clasificador de Issues")
    print("=" * 60)
    
    # Validar tokens
    if not SONAR_TOKEN:
        print("ERROR: No se encontro API_SONAR ni SONAR_TOKEN en variables de entorno", file=sys.stderr)
        sys.exit(1)
    if not NVIDIA_API_KEY:
        print("ERROR: No se encontro API_NVIDIA ni NVIDIA_API_KEY en variables de entorno", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n[1/3] Obteniendo issues de SonarCloud (proyecto: {PROJECT_KEY})...")
    issues = get_sonar_issues()
    if not issues:
        print("No se encontraron issues abiertos. Usando datos de ejemplo del informe...")
        issues = [
            {'id': '1', 'title': 'Replace this assertion to not have the same actual and expected expression', 'file': 'tests/test_requests.py', 'severity': 'Major', 'type': 'Bug'},
            {'id': '2', 'title': 'Replace this assertion to not have the same actual and expected expression', 'file': 'tests/test_requests.py', 'severity': 'Major', 'type': 'Bug'},
            {'id': '3', 'title': 'Replace this assertion to not have the same actual and expected expression', 'file': 'tests/test_requests.py', 'severity': 'Major', 'type': 'Bug'},
            {'id': '4', 'title': 'Add the "autouse" marker to this fixture', 'file': 'tests/test_utils.py', 'severity': 'Major', 'type': 'Bug'},
            {'id': '5', 'title': 'Add a "title" attribute to this iframe', 'file': 'docs/_templates/sidebar.html', 'severity': 'Minor', 'type': 'Bug'},
        ]
    
    print(f"   Issues encontrados: {len(issues)}")
    modules = get_modules(issues)
    
    print(f"\n[2/3] Enviando a SQA-Triage (NVIDIA NIM - {NVIDIA_MODEL})...")
    result = call_nvidia(issues, modules)
    
    print(f"\n[3/3] Guardando resultado...")
    output = f"""# SQA-Triage Report
## Ejecucion Automatica - {NVIDIA_MODEL}
**Fecha:** $(date -u)
**Proyecto:** {PROJECT_KEY}
**Issues analizados:** {len(issues)}

---

{result}

---

*Reporte generado automaticamente por SQA-Triage Agent.*
*[IA] Esta clasificacion fue producida con asistencia de un modelo de lenguaje (NVIDIA NIM).*
*Cada salida debe ser verificada por un ingeniero humano antes de incluirse en el informe.*
"""
    
    with open('triage_result.md', 'w') as f:
        f.write(output)
    
    print(f"\n   Resultado guardado en triage_result.md")
    print(f"\n{'=' * 60}")
    print("TRIAJE COMPLETADO")
    print(f"{'=' * 60}")
    print(result)


if __name__ == '__main__':
    main()
