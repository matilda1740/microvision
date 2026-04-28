# Case Study: Scalable Dependency Discovery with MicroVision

## 🎯 The Mission
The goal was to reconstruct the architecture of a complex OpenStack deployment using only its raw system logs. 

**Constraints**:
*   No codebase access.
*   No distributed tracing (Zipkin/Jaeger).
*   No prior knowledge of service endpoints.

## 🛠️ The Implementation

### 1. Handling Scale (183k Logs)
Standard log parsers often fail on "Template Bloat"—where every unique UUID creates a new template. I customized the **Drain3** parser to mask high-entropy variables (IPs, Request IDs) during the extraction phase, reducing the template pool from 25,000 to a manageable **1,200 core behaviors**.

### 2. Semantic Retrieval
Instead of strict string matching, I mapped log templates to vectors. This allowed the system to understand that:
*   `"Starting image upload for ID: 123"`
*   `"Received GLANCE request to store binary"`
...are semantically related even if they share zero shared keywords.

### 3. The "LLM-as-a-Judge" Filter
A known issue in log analysis is **temporal noise**: two logs appearing together in time doesn't prove dependency.
I implemented a validation layer that prompts an LLM with two templates and asks: *"Based on the technical context of these logs, is there a likely causal relationship?"* 

**Result**: We discarded 42% of candidate edges that were merely coincidental, boosting Precision from 38% to 54%.

## 📈 Results & Impact

*   **Automation**: Reduced the time to map a new microservice environment from days to minutes.
*   **Accuracy**: Achieved a 0.54 Precision score on the OpenStack benchmark, outperforming syntactic baselines.
*   **Defense Grade**: The "Alpha Slider" in the UI allows operators to tune the sensitivity of the discovery engine in real-time.

## 🧠 Key Learnings
*   **Vector DB Selection**: ChromaDB was chosen for its zero-config setup, crucial for rapidly prototyping semantic pipelines.
*   **Prompt Engineering**: Context is everything. Providing the LLM with the *service names* as well as the *log content* significantly improved validation accuracy.

---
### 📖 Deep Dive: [Technical Architecture & Results](TECHNICAL_DETAILS.md)
*For a detailed breakdown of the Data Flow, Hybrid Retrieval (Dense + Sparse), and full Metric Analysis.*
