# Multimodal RAG System for Manufacturing Operations Equipement Maintenance

**Course:** Multimodal Retrieval-Augmented Generation Bootcamp (BITS Pilani WILP)  
**Submission Type:** Individual Assignment  

---

## 1. Problem Statement

**Domain Identification**
This project is situated within the domain of Manufacturing Operations and Automation Engineering, specifically focusing on the management, retrieval, and application of operational paperwork, equipment documentation, and maintenance procedures to reduce Mean Time To Repair (MTTR) for breakdown equipment.

**Problem Description**
Modern manufacturing facilities rely on highly automated production lines spread across multiple shops such as Engine, Transaxle, Press, BIW, Paint, and Assembly. These shops use a wide variety of equipment including Programmable Logic Controllers (PLCs), robotic manipulators, servo systems, conveyors, distributed control systems, sensors, actuators, and machine controllers sourced from multiple different vendors. In such an environment, the number of installed equipment assets can run into 3000+ units, with an even larger number of manuals, OEM documents, troubleshooting guides, safety procedures, maintenance records, and operating instructions.
Keeping this machinery running optimally requires maintenance engineers and shop-floor technicians to navigate an overwhelming volume of operational paperwork. The challenge is not just the volume, but also the format of the information. Equipment documentation is highly dense, multimodal, and continuously evolving. During a machine fault or breakdown, an engineer may need to quickly cross-reference text-based troubleshooting instructions, configuration tables, alarm code descriptions, wiring diagrams, pneumatic circuits, or mechanical schematics to identify the root cause and restore the machine.
Traditional keyword-based search mechanisms are not adequate in this setting. They may return broad matches for a term, but they are unable to effectively interpret important information embedded inside PDF tables, scanned documents, or technical diagrams. As a result, maintenance personnel often spend valuable time searching through multiple manuals to find the exact page or parameter relevant to the breakdown. In a high-volume manufacturing setup, this delay directly increases equipment downtime and impacts production continuity, where even short interruptions can result in significant operational losses.


**Why This Problem Is Unique**
What distinguishes manufacturing equipment maintenance from generic document retrieval is the combination of technical complexity, frequent change, and the large number of equipment types operating in the same environment. Production lines are regularly upgraded, modified, or integrated with newer components, while some legacy machines continue to operate alongside modern automation systems. An engineer may therefore encounter a situation where an older servo drive or PLC is interfacing with a newly introduced gateway, sensor, or controller.
The associated documentation reflects this complexity. It contains highly specific alphanumeric part numbers and model identifiers such as Siemens S7-1500, Omron E2E-X5, vendor-specific parameter codes, electrical tolerances, alarm descriptions, and equipment-specific safety conditions. In addition, the visual content in these manuals is not decorative — it is functionally critical. A P&ID, ladder logic diagram, wiring schematic, pneumatic circuit, or interlock flowchart may contain key operational states or diagnostic logic needed for troubleshooting.
This creates a problem that is fundamentally multimodal. Standard dense vector retrieval may capture semantic similarity in text, but it often struggles with exact alphanumeric matching, fault codes, or structured engineering references. At the same time, a pure keyword-based system cannot understand the contextual meaning of tables and visual layouts. Without a robust multimodal retrieval pipeline, the critical information hidden inside calibration tables, circuit drawings, and OEM troubleshooting pages remains difficult to access at the exact moment it is needed.


**Why RAG Is the Right Approach**
A Multimodal Retrieval-Augmented Generation (RAG) system is well suited to handle the strict requirements and rapidly changing nature of manufacturing documentation. Alternative approaches, such as fine-tuning a Large Language Model (LLM) directly on factory manuals, are impractical and can introduce serious risks. In equipment maintenance, inaccurate responses involving voltage limits, torque values, clearances, sensor settings, or recovery steps can lead to equipment damage, repeat failures, or unsafe interventions. A system used in this domain must therefore minimize hallucination and ensure that responses are grounded in actual source documents.
RAG addresses this challenge by separating reasoning from stored knowledge. Instead of relying on the model’s memory, the system first retrieves the most relevant content from indexed manuals and then generates an answer based only on those retrieved chunks. By parsing text, tables, and diagram summaries into a hybrid searchable index, the solution can dynamically access the latest available documentation without requiring repeated model retraining whenever manuals are updated or equipment versions change.
This approach also provides traceability. The answer can point the user to the exact supporting page or section of the equipment manual, allowing maintenance personnel to verify the recommendation before taking action. This is especially valuable in breakdown situations, where quick access to the correct document and precise information can significantly improve troubleshooting efficiency and reduce MTTR.


**Expected Outcomes**
A successful implementation of this Multimodal RAG system will help reduce Mean Time To Repair (MTTR) and improve operational efficiency for maintenance teams working across multiple shops in an automobile manufacturing plant. Instead of manually opening several manuals and searching page by page, a maintenance engineer or floor technician will be able to ask a natural-language question and receive a precise, grounded, and cited response based on the most relevant source documents.
For example, a user may ask:
"The assembly line PLC is throwing error code E-704; based on the latest OEM table, what is the required sensor calibration, and where is the safety reset relay located in the schematic?"
In such a case, the system should retrieve:
•	the textual explanation of the fault,
•	the relevant row from the calibration or parameter table,
•	and the summarized description of the associated wiring or control schematic.
These retrieved elements can then be synthesized into a precise response that helps the maintenance team understand the reason for breakdown and locate the correct corrective reference quickly. By reducing the effort required to search through multiple equipment manuals and by directing the user to the exact relevant document section, the system improves troubleshooting speed, enhances confidence in maintenance decisions, and supports safer and more efficient plant operations.

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


## 3. Architecture Overview
