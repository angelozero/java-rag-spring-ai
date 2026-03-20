package com.angelozero.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.nio.file.Path;
import java.nio.file.Paths;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;

@Component
public class IngestionConfig {


    @EventListener(ApplicationReadyEvent.class)
    public void startIngestion() {
        try {
            System.out.println("Iniciando ingestão do documento...");

            var resource = getClass().getClassLoader().getResource("docs/article_thebeatoct2024.pdf");

            if (resource == null) {
                throw new RuntimeException("Arquivo não encontrado na pasta src/main/resources/docs/");
            }

            Path path = Paths.get(resource.toURI());

            // Load document (PDF, TXT, etc.)
            Document document = loadDocument(path, new ApachePdfBoxDocumentParser());

            // Split document into smaller chunks
            // 300 tokens per chunk, 50 tokens overlap for context continuity
            DocumentSplitter splitter = DocumentSplitters.recursive(300, 50);

            // Create embedding model (384 dimensions for AllMiniLmL6V2)
            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

            // Create pgvector embedding store
            EmbeddingStore<TextSegment> embeddingStore = PgVectorEmbeddingStore.builder()
                    .host("localhost")
                    .port(5432)
                    .database("postgres")
                    .user("user")
                    .password("pass")
                    .table("document_embeddings")
                    .dimension(embeddingModel.dimension())  // 384 for AllMiniLmL6V2
                    .build();

            // Ingest: split document, generate embeddings, and store in pgvector
            EmbeddingStoreIngestor.builder()
                    .documentSplitter(splitter)
                    .embeddingModel(embeddingModel)
                    .embeddingStore(embeddingStore)
                    .build()
                    .ingest(document);

            System.out.println("Document ingested successfully!");

        } catch (Exception ex) {
            System.err.println("ERRO NA INGESTÃO: " + ex.getMessage());
        }
    }
}