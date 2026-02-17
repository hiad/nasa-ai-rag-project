
## 📝 Submission Guidelines

When submitting your completed project:
1. Ensure all TODO items are implemented
2. Test the complete workflow end-to-end
3. Include a brief report on challenges faced and solutions found (see below)
4. Document any additional features or improvements you added

## 📝 Project Report: Challenges & Solutions

### 1. Precision in RAG Retrieval
*   **Challenge**: We observed lower "Answer Relevancy" scores (around 0.56) because the LLM was being too conversational or adding general knowledge not found in the documents.
*   **Solution**: 
    *   **Prompt Engineering**: Updated the system prompt to include a strict "DIRECTIVE" to only use provided context.
    *   **Model Upgrade**: Try different models to find the best one for the task.
    *   **Metadata Filtering**: Implemented ChromaDB `where` filtering to allow users to narrow down searches to specific missions.


## 📋 QUERY SAMPLE

To verify your RAG system, you can use the following sample queries. These responses are based on the actual documents in the `data_text` directory.

### 1. Apollo 11 Technical Documentation
*   **Query**: "What is document MPR-SAT-FE-69-9 about?"

### 2. Challenger (STS-51L) Launch Delay
*   **Query**: "Why was the Challenger launch postponed for 24 hours on Saturday night?"

### 3. Apollo 13 Hardware
*   **Query**: "What specific equipment was used to record the AS13_CM voice transcription?"

### 4. Challenger Crew Schedule
*   **Query**: "At what time was the Challenger crew scheduled to be awakened on launch day?"

---