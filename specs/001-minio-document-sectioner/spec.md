## Goal

Extend the existing `minio-document-sectioner` service so it can participate in an asynchronous document-processing pipeline coordinated by the backend, Celery, and MinIO.

The backend owns the original document and its initially extracted files. When `main.txt` becomes available, the backend must notify this service through an HTTP endpoint. The service must immediately create an asynchronous Celery job and return:

```json
{
  "job-id": "generated-job-id",
  "start-time": "ISO-8601-timestamp"
}
```

The request must not wait for document processing to finish.

The worker must download `main.txt` from the document’s MinIO directory, process and section its content using the configured classification pipeline, generate all required outputs, and upload those outputs into the same document directory in MinIO.

OCR processing is a separate stage. When `OCR.txt` becomes available, the backend must call a second endpoint. That endpoint must create another asynchronous Celery job, return its own job ID and start time immediately, process the OCR content, generate or update the relevant sectioning outputs, upload them into the document directory, and record a structured success or failure log.
Integrate the existing minio-document-sectioner with the backend, Celery, and MinIO as an asynchronous two-stage document-processing service.
The backend notifies the service when main.txt or OCR.txt becomes available. The service must immediately return a job ID and start time, process the requested stage asynchronously, classify document sections using Laya Multilingual with an LLM fallback, and upload all generated outputs and structured job logs into the same document directory in MinIO.

The existing classification strategy must remain hybrid:

* Use `convaiinnovations/laya-multilingual` as the primary multilingual section classifier.
* Use the configured LLM only as a fallback when Laya’s result is missing, invalid, ambiguous, unsupported, or below the configured confidence threshold.
* Preserve deterministic output formats and document ordering.
* Keep backend-owned source objects unchanged.
* Make every processing stage idempotent, observable, retry-safe, and suitable for production deployment.

## Details

### 1. MinIO document structure

The service must work with the following document structure:

```text
media/
└── documents/
    └── {document_id}/
        ├── original.pdf
        ├── main.txt
        ├── images/
        │   ├── page-001.jpg
        │   ├── page-002.jpg
        │   └── ...
        ├── OCR/
        │   ├── page-001.json.txt
        │   ├── page-001-text.txt
        │   ├── page-002.json.txt
        │   ├── page-002-text.txt
        │   └── ...
        ├── Figure-Table/
        │   ├── page-001.json.txt
        │   ├── page-001-table-001.png
        │   ├── page-001-figure-001.png
        │   └── ...
        ├── Stamp-Signature/
        │   ├── page-001.json.txt
        │   ├── page-001-stamp-001.png
        │   ├── page-001-signature-001.png
        │   └── ...
        └── OCR.txt
```

All paths must be constructed safely from `document_id`. Reject values containing path traversal, unexpected separators, empty values, or unsupported characters.

The service must never overwrite or delete these backend-owned inputs:

* `original.pdf`
* `main.txt`
* `images/*`

The service must also avoid modifying AI-module source outputs unless modification is explicitly part of the sectioner’s ownership contract:

* `OCR/*`
* `Figure-Table/*`
* `Stamp-Signature/*`
* `OCR.txt`

Sectioner-generated files must be stored under a dedicated directory inside the same document folder:

```text
media/documents/{document_id}/Sectioning/
```

Recommended structure:

```text
Sectioning/
├── main/
│   ├── sections.json
│   ├── sections.txt
│   ├── sections.docx
│   ├── manifest.json
│   └── processing-summary.json
├── ocr/
│   ├── sections.json
│   ├── sections.txt
│   ├── sections.docx
│   ├── manifest.json
│   └── processing-summary.json
├── jobs/
│   └── {job_id}/
│       ├── success.json
│       └── failure.json
└── latest.json
```

Only one of `success.json` or `failure.json` should normally exist for a completed job.

### 2. Main-text-ready endpoint

Provide an endpoint similar to:

```text
POST /api/v1/documents/main-ready
```

Example request:

```json
{
  "document-id": "document-123"
}
```

The endpoint must:

1. Validate the request.
2. Verify that the document directory exists.
3. Verify that `main.txt` exists in MinIO.
4. Read the object metadata without processing the full document inside the API request.
5. Generate a unique job ID.
6. Save the initial job state.
7. enqueue a Celery task for the `main` stage.
8. Return HTTP `202 Accepted` immediately.

Required response:

```json
{
  "job-id": "01JMAINEXAMPLE",
  "start-time": "2026-09-23T10:30:00Z"
}
```

`start-time` must be an ISO 8601 UTC timestamp.

The endpoint must not perform model inference, document sectioning, DOCX generation, or large MinIO downloads before returning the response.

### 3. OCR-ready endpoint

Provide a separate endpoint similar to:

```text
POST /api/v1/documents/ocr-ready
```

Example request:

```json
{
  "document-id": "document-123"
}
```

This endpoint means that the backend has completed OCR processing and `OCR.txt` should now be available.

The endpoint must:

1. Validate the document ID.
2. Verify that `OCR.txt` exists.
3. Optionally verify the presence and consistency of per-page OCR files.
4. Create a new job for the `ocr` stage.
5. enqueue the OCR Celery task.
6. Return HTTP `202 Accepted` immediately.

Required response:

```json
{
  "job-id": "01JOCREXAMPLE",
  "start-time": "2026-09-23T10:45:00Z"
}
```

The OCR job must not be executed as part of the main-ready API request. Main processing and OCR processing are independent asynchronous stages.

### 4. Job model

Every job must contain at least:

* Unique job ID
* Document ID
* Processing stage: `main` or `ocr`
* Current status
* Creation time
* Start time
* Completion time
* Source MinIO bucket
* Source object key
* Source object ETag or checksum
* Retry count
* Error code
* Sanitized error message
* Generated output paths
* Model information
* Processing statistics

Supported job states should include:

```text
queued
running
uploading
verifying
succeeded
partially_succeeded
failed
retrying
cancelled
```

Valid transitions must be enforced. For example:

```text
queued → running → uploading → verifying → succeeded
```

A failed transient operation may transition through `retrying`, while a permanent validation error should transition directly to `failed`.

### 5. Job-status endpoint

Provide an endpoint similar to:

```text
GET /api/v1/jobs/{job_id}
```

Example successful response:

```json
{
  "job-id": "01JMAINEXAMPLE",
  "document-id": "document-123",
  "stage": "main",
  "status": "succeeded",
  "start-time": "2026-09-23T10:30:00Z",
  "completed-time": "2026-09-23T10:31:42Z",
  "outputs": [
    "media/documents/document-123/Sectioning/main/sections.json",
    "media/documents/document-123/Sectioning/main/sections.txt",
    "media/documents/document-123/Sectioning/main/sections.docx",
    "media/documents/document-123/Sectioning/main/manifest.json"
  ],
  "error": null
}
```

Do not expose internal stack traces, credentials, MinIO secrets, model-provider credentials, or raw infrastructure errors through this endpoint.

### 6. Main-stage processing

The main Celery task must:

1. Load the job record.
2. Acquire a document-stage lock.
3. verify that `main.txt` still exists.
4. Compare its current ETag or checksum with the value registered when the job was created.
5. Download the file into a job-specific temporary workspace.
6. Detect and normalize its text encoding.
7. Parse the text into paragraphs or logical blocks.
8. Preserve the original paragraph ordering.
9. Classify the blocks.
10. Assemble the detected document sections.
11. Generate JSON, TXT, DOCX, manifest, and summary outputs.
12. Upload outputs to the `Sectioning/main/` directory.
13. Verify that uploaded objects exist and match the expected size or checksum.
14. Write the structured success log.
15. Update the job state to `succeeded`.
16. Release the lock and safely remove temporary files.

If an unrecoverable error occurs, the task must update the job state and write a structured failure log.

### 7. OCR-stage processing

The OCR Celery task must:

1. Verify and download `OCR.txt`.
2. Discover available files under `OCR/`.
3. Sort page files by their numeric page number.
4. Use `page-NNN-text.txt` as the clean page-level OCR text when available.
5. Use `page-NNN.json.txt` for OCR metadata, provenance, confidence, bounding boxes, polygons, and raw response data when needed.
6. Preserve page ordering.
7. Detect missing or duplicated page numbers.
8. Process OCR text without modifying its source files.
9. Generate OCR-based sections and document outputs.
10. Add page references to section metadata when they can be determined.
11. Upload results to `Sectioning/ocr/`.
12. Verify the uploaded objects.
13. Write a structured success or failure log.

The OCR job may reference related assets in `Figure-Table/` and `Stamp-Signature/`, but it must not assume that every page contains those assets.

A missing optional figure, table, stamp, or signature must not cause the entire job to fail.

### 8. Classification behavior

Use `convaiinnovations/laya-multilingual` as the primary classification model.

The model must be loaded once per worker process and reused between jobs. It must not be loaded again for every paragraph.

Expected configuration:

```text
CLASSIFIER_BACKEND=hybrid
LAYA_MODEL_ID=convaiinnovations/laya-multilingual
LAYA_DEVICE=cuda
LAYA_REQUIRE_CUDA=true
LAYA_CONFIDENCE_THRESHOLD=0.85
LAYA_MAX_PARAGRAPH_CHARS=12000
```

The configured LLM acts only as a fallback.

Use the fallback when:

* Laya confidence is below the threshold.
* Laya returns no valid label.
* The result cannot be mapped to a supported section type.
* The input is ambiguous.
* The model output is malformed.
* A block contains mixed section types that require additional reasoning.

Do not send every paragraph to both models.

Record the selected classifier for each section:

```text
laya
llm-fallback
rule
```

Also record the confidence score where available.

Long inputs must be split according to the actual tokenizer limits. Do not depend exclusively on character count. Prefer splitting at paragraph, sentence, or line boundaries. Preserve the original ordering and aggregate chunk-level predictions into a final section decision.

### 9. Output JSON

`sections.json` must contain document-level metadata and ordered sections.

Recommended structure:

```json
{
  "schema-version": "1.0",
  "document-id": "document-123",
  "job-id": "01JMAINEXAMPLE",
  "stage": "main",
  "source": {
    "bucket": "configured-bucket",
    "object-key": "media/documents/document-123/main.txt",
    "etag": "source-etag"
  },
  "models": {
    "primary": "convaiinnovations/laya-multilingual",
    "fallback": "configured-llm-model"
  },
  "sections": [
    {
      "section-id": "section-001",
      "type": "introduction",
      "title": "Introduction",
      "text": "Section content",
      "order": 1,
      "page-start": null,
      "page-end": null,
      "classifier": "laya",
      "confidence": 0.93,
      "source-blocks": [1, 2, 3]
    }
  ]
}
```

The exact supported section labels must be documented and normalized consistently.

Unknown classifications must use an explicit value such as `unknown` or `other`; they must not be silently discarded.

### 10. TXT and DOCX outputs

`sections.txt` must contain a readable, ordered representation of all detected sections.

It should include:

* Section title
* Normalized section type
* Section text
* Page range when available
* Clear separators between sections

`sections.docx` must contain the same logical content with appropriate Word headings and readable paragraph formatting.

The DOCX file must not include secrets, debug traces, raw prompts, or internal model responses.

### 11. Manifest

Each stage must create `manifest.json`.

The manifest must include:

* Document ID
* Job ID
* Stage
* Source object path
* Source ETag or checksum
* Processing start and completion times
* Application version
* Schema version
* Primary model
* Fallback model
* Confidence threshold
* Generated object paths
* Object sizes
* Object checksums when available
* Warning count
* Error count
* Number of processed blocks
* Number of generated sections
* Number of fallback calls
* Final job status

The manifest should be uploaded only after the other required outputs have been uploaded and verified. It acts as the completion marker for that processing stage.

### 12. Processing summary

Create `processing-summary.json` containing operational statistics without exposing sensitive data.

Include:

* Total processing duration
* MinIO download duration
* Classification duration
* Output-generation duration
* Upload duration
* Input character count
* Input token estimate
* Paragraph or block count
* Section count
* Laya prediction count
* LLM fallback count
* Low-confidence count
* Warning list
* Missing optional asset list

### 13. Success log

A success log must only be written after all required outputs have been uploaded and verified.

Recommended path:

```text
media/documents/{document_id}/Sectioning/jobs/{job_id}/success.json
```

Example content:

```json
{
  "event": "document-sectioning-succeeded",
  "job-id": "01JMAINEXAMPLE",
  "document-id": "document-123",
  "stage": "main",
  "status": "succeeded",
  "start-time": "2026-09-23T10:30:00Z",
  "completed-time": "2026-09-23T10:31:42Z",
  "source-object": "media/documents/document-123/main.txt",
  "source-etag": "source-etag",
  "output-count": 5,
  "warnings": []
}
```

Writing this object is part of successful job completion.

### 14. Failure log

For permanent failures, write:

```text
media/documents/{document_id}/Sectioning/jobs/{job_id}/failure.json
```

It must include:

* Job ID
* Document ID
* Stage
* Failure time
* Safe error code
* Sanitized error message
* Retry count
* Source object path
* Source ETag when available
* Last successfully completed operation
* Whether partial output files were created

Do not upload secrets, access keys, authorization headers, full stack traces, or raw LLM credentials.

### 15. Idempotency

Requests and tasks must be idempotent.

Use a processing identity based on:

```text
document_id + stage + source ETag/checksum + pipeline version
```

If an identical request has already succeeded, return or reference the existing successful job instead of processing it again.

If an identical job is currently queued or running, return the existing active job.

Create a new job if:

* The source object changed.
* The source ETag changed.
* The processing pipeline version changed.
* A forced reprocessing option was explicitly requested and authorized.

Duplicate backend notifications must not create duplicate output sets or simultaneous processing for the same document and stage.

### 16. Locking and concurrency

Use a distributed lock for each:

```text
document_id + stage
```

The lock must have a safe expiration and ownership token.

Only the worker that owns the lock may release it.

Main and OCR jobs may have separate locks, but output publishing must remain consistent. Two workers must never concurrently overwrite the same canonical output directory for the same document and stage.

### 17. Atomic output publishing

Generate files in a job-specific temporary workspace.

Upload intermediate objects to a job-specific staging prefix when needed:

```text
media/documents/{document_id}/Sectioning/.staging/{job_id}/
```

After every required object has been generated successfully:

1. Upload the objects.
2. Verify size or checksum.
3. Publish them to the canonical stage directory.
4. Upload `manifest.json` last.
5. Update `latest.json`.
6. Write `success.json`.

Consumers must treat the presence of a valid manifest as the signal that a complete output set is ready.

A failed job must not leave a new manifest pointing to incomplete outputs.

### 18. Retry behavior

Retry transient failures such as:

* Temporary MinIO connectivity errors
* Network timeouts
* Temporary database failures
* Celery broker interruptions
* Temporary LLM-provider errors
* Rate limits
* Recoverable GPU allocation errors

Use exponential backoff with jitter and a configurable retry limit.

Do not retry permanent failures such as:

* Invalid document ID
* Missing required source object after the allowed consistency window
* Empty required input
* Unsupported encoding that cannot be decoded
* Invalid configuration
* Invalid MinIO credentials
* Permanently malformed input

Celery tasks should use late acknowledgements and must be safe to execute again after worker termination.

### 19. MinIO consistency handling

Immediately after receiving a ready notification, the object may still be settling.

The service should:

1. Check object existence and metadata.
2. Wait for a short configurable consistency interval if necessary.
3. Read metadata again.
4. Confirm that size and ETag are stable.
5. Start processing only after stability is confirmed.

If the object does not appear within the configured timeout, mark the job as failed with a clear source-not-found error.

### 20. Partial success

Use `partially_succeeded` only when required outputs are complete but one or more optional enrichments failed.

Examples:

* A figure reference could not be associated with a section.
* One optional OCR metadata file was malformed.
* A stamp image was missing.
* A nonessential DOCX enrichment failed while the required JSON and TXT outputs remain valid, if this behavior is explicitly allowed by configuration.

Missing or corrupt required outputs must result in `failed`, not `partially_succeeded`.

### 21. Figure and table references

For OCR-stage processing, inspect `Figure-Table/page-NNN.json.txt` when available.

Section metadata may contain references such as:

```json
{
  "figures": [
    {
      "page": 2,
      "object-key": "media/documents/document-123/Figure-Table/page-002-figure-001.png"
    }
  ],
  "tables": [
    {
      "page": 2,
      "object-key": "media/documents/document-123/Figure-Table/page-002-table-001.png"
    }
  ]
}
```

Do not duplicate or rewrite the source images.

Ignore untrusted absolute paths contained inside module-generated JSON. Resolve assets only from the permitted document prefix.

### 22. Stamp and signature references

Inspect `Stamp-Signature/page-NNN.json.txt` when available.

A section may reference related stamp or signature images using their MinIO object keys.

These assets are optional. Their absence must be recorded as a warning only when the related JSON explicitly says that the asset should exist.

### 23. Page ordering

Page numbers must be parsed numerically.

For example, the correct order is:

```text
page-001
page-002
page-010
page-011
```

Do not rely on unsafe or inconsistent filename ordering.

Detect and report:

* Missing page numbers
* Duplicate page numbers
* Invalid page filenames
* OCR pages that have JSON but no text
* OCR pages that have text but no JSON

Processing should continue when missing page data is noncritical.

### 24. Latest document status

Maintain:

```text
media/documents/{document_id}/Sectioning/latest.json
```

It should summarize the latest known result of both stages:

```json
{
  "document-id": "document-123",
  "main": {
    "job-id": "01JMAINEXAMPLE",
    "status": "succeeded",
    "manifest": "media/documents/document-123/Sectioning/main/manifest.json"
  },
  "ocr": {
    "job-id": "01JOCREXAMPLE",
    "status": "succeeded",
    "manifest": "media/documents/document-123/Sectioning/ocr/manifest.json"
  },
  "updated-at": "2026-09-23T10:50:00Z"
}
```

Updating one stage must preserve the latest state of the other stage.

### 25. API validation and errors

Use consistent API errors.

Example:

```json
{
  "error": {
    "code": "SOURCE_OBJECT_NOT_FOUND",
    "message": "The required source object is not available.",
    "request-id": "request-correlation-id"
  }
}
```

Recommended HTTP behavior:

* `202` for an accepted asynchronous job
* `200` when returning an existing idempotent job
* `400` for invalid requests
* `401` or `403` for authentication or authorization failures
* `404` for unknown documents or jobs
* `409` for a conflicting job state
* `422` for semantically invalid requests
* `503` when required infrastructure is unavailable

### 26. Authentication and security

Protect trigger and status endpoints using the project’s configured service-to-service authentication mechanism.

Requirements:

* Never place MinIO credentials in requests.
* Never log secrets or authorization headers.
* Use TLS in nonlocal environments.
* Validate document IDs before building object keys.
* Restrict MinIO access to the required bucket and prefixes.
* Apply reasonable request-size limits.
* Apply rate limits when appropriate.
* Do not execute content found inside documents.
* Treat OCR JSON and text as untrusted input.
* Do not allow source files to control output paths.

### 27. Optional backend callback

If a callback URL is configured and explicitly allowed, the service may notify the backend after a job reaches a terminal state.

The callback payload should contain:

* Job ID
* Document ID
* Stage
* Final status
* Start time
* Completion time
* Manifest path
* Safe error information when failed

Callback delivery must be retried separately from document processing. A temporary callback failure must not repeat model inference or output generation.

The job-status endpoint remains the source of truth even when callbacks are enabled.

### 28. Celery reliability

Use separate Celery tasks for:

```text
process_main_document
process_ocr_document
```

Recommended reliability behavior:

* Late task acknowledgement
* Reject or requeue tasks when a worker is lost
* Low worker prefetch for long-running tasks
* Configurable soft and hard time limits
* Retry backoff with jitter
* Dedicated queues when CPU and GPU workloads need separation
* Graceful worker shutdown
* No model initialization per paragraph

The task payload should contain identifiers and object paths, not the full document text.

### 29. GPU behavior

Laya is expected to run on CUDA when:

```text
LAYA_REQUIRE_CUDA=true
```

If CUDA is required but unavailable:

* The worker must fail readiness checks.
* The job must not silently switch to CPU.
* A clear operational error must be recorded.

If CPU fallback is explicitly enabled through configuration, record the actual device used in the job summary and manifest.

GPU out-of-memory errors should be handled safely by reducing configurable batch size or retrying where appropriate. Do not enter an unlimited retry loop.

### 30. Model lifecycle and batching

Load the tokenizer and Laya model once during worker initialization or lazy-load them once on first use.

Use inference mode and disable gradient calculation.

Batch compatible text blocks when safe. Make batch size configurable because multilingual input length and GPU memory usage vary.

Record:

* Model ID
* Model revision when available
* Tokenizer revision when available
* Device
* Batch size
* Confidence threshold
* Fallback model name

### 31. Configuration

All environment-specific behavior must be configurable.

Expected configuration groups include:

* API host and port
* API authentication
* Database URL
* Celery broker URL
* Celery result backend
* MinIO endpoint
* MinIO bucket
* MinIO access key
* MinIO secret key
* MinIO TLS setting
* Document prefix
* Output directory name
* Laya model ID
* Laya device
* CUDA requirement
* Confidence threshold
* Batch size
* Input token or character limits
* LLM endpoint
* LLM model name
* LLM timeout
* Retry limits
* Task time limits
* Object-settle timeout
* Lock timeout
* Log level
* Callback configuration

Secrets must not have insecure production defaults.

### 32. Observability

Use structured logs containing:

* Request ID
* Job ID
* Document ID
* Stage
* Task ID
* Worker name
* Event name
* Duration
* Retry number
* Final status

Do not log complete document text.

Expose health endpoints similar to:

```text
GET /health/live
GET /health/ready
```

Liveness should confirm that the process is alive.

Readiness should verify the dependencies required by that service, such as configuration validity, database connectivity, broker connectivity, MinIO access, and GPU/model readiness where appropriate.

### 33. Metrics

Provide metrics where the existing project supports them.

Useful metrics include:

* Jobs accepted by stage
* Jobs succeeded by stage
* Jobs failed by stage
* Active jobs
* Retry count
* Processing duration
* MinIO download and upload duration
* Laya inference duration
* LLM fallback duration
* Fallback rate
* Low-confidence rate
* Processed character or token count
* GPU memory failures
* Output verification failures

Metrics must not include raw document text or high-cardinality object content.

### 34. Testing expectations

Cover at least the following scenarios:

* Valid main-ready request
* Valid OCR-ready request
* Immediate `202` response
* Missing `main.txt`
* Missing `OCR.txt`
* Empty source file
* Duplicate ready notification
* Two concurrent requests for the same document and stage
* Same document with a changed ETag
* Laya high-confidence result
* Laya low-confidence result using LLM fallback
* Invalid Laya output
* Temporary LLM failure
* MinIO download failure
* MinIO upload failure
* Verification failure
* Worker retry
* Worker termination followed by task redelivery
* Correct page ordering
* Missing optional OCR assets
* Malformed per-page OCR JSON
* Successful TXT, JSON, and DOCX creation
* Manifest uploaded last
* Success log written only after verification
* Failure log with sanitized information
* CUDA-required worker without CUDA
* Job-status endpoint
* Prevention of path traversal
* Preservation of backend-owned files

Tests must mock external services where appropriate and include an integration path using test containers or equivalent infrastructure when supported.

### 35. Completion criteria

The work is considered complete when:

* The backend can notify the service that `main.txt` is ready.
* The service immediately returns `job-id` and `start-time`.
* Main processing executes asynchronously through Celery.
* The backend can independently notify the service that `OCR.txt` is ready.
* OCR processing executes asynchronously through a separate Celery task.
* Both stages retrieve their inputs from MinIO.
* Laya is the primary classifier.
* The configured LLM is used only as a controlled fallback.
* Generated outputs are stored inside the same document’s `Sectioning/` directory.
* Backend-owned inputs remain unchanged.
* Jobs are idempotent and retry-safe.
* Job status can be queried.
* Required output objects are verified before success is recorded.
* Structured success and failure logs are stored in MinIO.
* Automated tests cover the main workflow and failure cases.
* Runtime configuration and the backend integration contract are clearly documented.
