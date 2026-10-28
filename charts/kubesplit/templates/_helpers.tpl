{{/* Shared snippets. Files starting with "_" render nothing on their own. */}}

{{- define "kubesplit.name" -}}
{{ .Chart.Name }}
{{- end }}

{{/* Release-scoped base name, e.g. "kubesplit" or "bench1-kubesplit". */}}
{{- define "kubesplit.fullname" -}}
{{- if contains .Chart.Name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/* Standard labels on every object, so kubectl/Grafana can filter by release. */}}
{{- define "kubesplit.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{ include "kubesplit.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/* The subset used to match pods. Must never change after install. */}}
{{- define "kubesplit.selectorLabels" -}}
app.kubernetes.io/name: {{ include "kubesplit.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "kubesplit.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}
{{- end }}

{{/* In-cluster DNS name clients use to reach the server's Service. */}}
{{- define "kubesplit.serverName" -}}
{{ include "kubesplit.fullname" . }}-server
{{- end }}
