# Multimodal RAG System for Manufacturing Operations Equipement Maintenance

**Course:** Multimodal Retrieval-Augmented Generation Bootcamp (BITS Pilani WILP)  
**Submission Type:** Individual Assignment  

---

## 1. Problem Statement

**Domain Identification**
This project is situated within the domain of Manufacturing Operations and Automation Engineering, specifically focusing on the management, retrieval, and application of operational paperwork, equipment documentation, and maintenance procedures to reduce Mean Time To Repair on breakdown equipments.

**Problem Description**
Modern manufacturing facilities rely on highly automated production lines, integrating diverse technologies such as Programmable Logic Controllers (PLCs), robotic manipulators, distributed control systems, and complex sensor networks from multiple different vendors. Keeping this machinery running optimally requires maintenance engineers and floor operators to navigate an overwhelming volume of operational paperwork. This includes Standard Operating Procedures (SOPs), Original Equipment Manufacturer (OEM) manuals, safety compliance documents, and maintenance logs. The core problem is that this paperwork is incredibly dense, highly multimodal, and constantly evolving. When a machine experiences a fault, an engineer must rapidly cross-reference text-based troubleshooting procedures with dense parameter configuration tables and intricate electrical or mechanical schematics (images). Traditional keyword search mechanisms are entirely inadequate.They are blind to the vital data locked inside PDF tables and cannot interpret visual diagrams, resulting in prolonged equipment downtime—a critical metric where every minute of delay can cost a facility lacs of rupees in lost production.

**Why This Problem Is Unique**
What distinguishes manufacturing equipment maintenance from generic document retrieval is the sheer volatility and technical density and number of equipments (3000+ nos) in the working environment. As production lines are optimized, there are a lot of components that keep updating, changing, or being swapped out for newer models. An engineer might be troubleshooting a legacy servo motor that is now interfacing with a newly installed IoT gateway. The paperwork reflects this chaotic reality, featuring highly specific, alphanumeric part numbers (e.g., "Siemens S7-1500" or "Omron E2E-X5"), complex parameter matrices, and strict safety tolerances. Furthermore, the visual data in this domain is strictly functional, not decorative. A Piping and Instrumentation Diagram (P&ID), a pneumatic circuit schematic, or a ladder logic flowchart contains vital operational states. Standard dense vector embeddings often fail at exact alphanumeric matching for part numbers, and without a robust multimodal pipeline, the rich, actionable data trapped in those schematics and calibration tables remains completely inaccessible to the user.

**Why RAG Is the Right Approach**
A Multimodal Retrieval-Augmented Generation (RAG) system is uniquely suited to handle the strict constraints and rapidly changing nature of manufacturing documentation. Alternative approaches, such as fine-tuning a Large Language Model (LLM) directly on factory manuals, are highly impractical and introduce severe safety risks. Fine-tuned models are prone to hallucinating critical numerical data—such as voltage limits, torque specifications, or safety clearances—which could lead to catastrophic equipment damage or worker injury. Furthermore, because automation components and their associated manuals are frequently updated to new versions, continuously retraining an LLM is cost-prohibitive and structurally rigid. RAG directly solves this by decoupling the reasoning engine from the underlying data. By parsing text, tables, and diagram summaries into a searchable vector and keyword hybrid index, the system dynamically retrieves only the most up-to-date, relevant documentation chunks.RAG grounds the LLM’s response in these explicit, verifiable sources, providing exact citations (e.g., referencing a specific page in a newly updated OEM manual) so engineers can independently verify the parameters before executing a physical repair

**Expected Outcomes**
A successful implementation of this Multimodal RAG system will drastically reduce Mean Time To Repair (MTTR) and improve operational efficiency on the factory floor.A floor operator or maintenance engineer will be able to query the system with complex, cross-modal questions in natural language[cite: 26].For example: *"The assembly line PLC is throwing error code E-704; based on the latest OEM table, what is the required sensor calibration, and where is the safety reset relay located in the schematic?"* The system will retrieve the text explanation of the error, extract the specific row from the calibration table, and pull the summarized description of the wiring diagram, synthesizing them into a precise, step-by-step resolution. By keeping a check on constantly updating components and multimodal manuals, the system ensures that engineers always have instant access to accurate, actionable, and cited operational knowledge, ultimately minimizing downtime and ensuring safe operations.

---

## 2. Architecture Overview

The system utilizes a hybrid local/cloud architecture designed for manufacturing environments. Document parsing, hybrid indexing (FAISS + BM25), and vector storage run locally, while heavy LLM/VLM reasoning is offloaded to APIs.

```mermaid
graph TD
    subgraph Ingestion Pipeline
        A[OEM Manuals / SOP PDFs] -->|PyMuPDF/Docling| B(Document Parser)
        B --> C{Chunk Type}
        C -->|Text/Paragraphs| D[Semantic Text Chunks]
        C -->|Markdown Tables| E[Parameter Table Chunks]
        C -->|Extracted Images| F[Vision Pre-Filter]
        
        F -->|Prompt: Reject Logos| G[Junk/Decorative]
        F -->|Prompt: Extract Data| H[P&ID / Wiring Summaries]
        G -->|Discarded| Z((Trash))
        
        D --> I[Local Embedding Model]
        E --> I
        H --> I
        
        I --> J[(FAISS HNSW Index)]
        I --> K[(BM25 Sparse Index)]
    end

    subgraph Query Pipeline
        L[Engineer Query] --> M[FastAPI /query]
        M --> N[Embedding Model]
        N --> O{Hybrid Retriever}
        
        O -->|Semantic| J
        O -->|Keyword/Part No.| K
        
        O --> P[Cross-Modal Context Merging]
        P --> Q[Generation LLM]
        Q --> R[Grounded Answer + Citations]
    end