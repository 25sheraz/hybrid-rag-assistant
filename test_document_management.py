import os
import sys
import unittest
from fastapi.testclient import TestClient
from dotenv import load_dotenv

load_dotenv()

from server import app
from services.document_service import DocumentIngestionService, DocumentMetadataStore

class TestDocumentManagement(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_filename = "test_policy_doc.txt"
        cls.test_content = (
            "DevelMo Document Management Security Policy.\n\n"
            "1. Security Compliance: All documents ingested into DevelMo RAG must be encrypted.\n"
            "2. Vector Indexing: Dual-vector dense and sparse BM25 indexing is required for enterprise search.\n"
            "3. Access Control: Document payload tags document_id must match authorization level."
        )

        # Cleanup existing documents in metadata store from previous runs
        service = DocumentIngestionService()
        for doc in service.list_documents():
            if doc.filename == cls.test_filename:
                service.delete_document(doc.id)


    def test_01_upload_document(self):
        print("\n--- Test 1: Uploading Document ---")
        files = {
            "file": (self.test_filename, self.test_content.encode("utf-8"), "text/plain")
        }
        response = self.client.post("/api/documents/upload", files=files)
        print("Upload Response Code:", response.status_code)
        print("Upload Response Data:", response.json())

        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("document", data)
        self.assertEqual(data["document"]["filename"], self.test_filename)
        self.assertGreater(data["document"]["chunk_count"], 0)

        # Store doc ID for subsequent tests
        TestDocumentManagement.uploaded_doc_id = data["document"]["id"]

    def test_02_duplicate_upload(self):
        print("\n--- Test 2: Testing Duplicate Document Upload ---")
        files = {
            "file": (self.test_filename, self.test_content.encode("utf-8"), "text/plain")
        }
        response = self.client.post("/api/documents/upload", files=files)
        print("Duplicate Response Code:", response.status_code)
        print("Duplicate Response Data:", response.json())

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("already been uploaded", data["error"].lower())
        self.assertEqual(data["existing_document_id"], getattr(TestDocumentManagement, "uploaded_doc_id", None))

    def test_03_list_documents(self):
        print("\n--- Test 3: Listing Uploaded Documents ---")
        response = self.client.get("/api/documents")
        print("List Response Code:", response.status_code)
        print("List Response Data:", response.json())

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("documents", data)
        doc_ids = [d["id"] for d in data["documents"]]
        self.assertIn(getattr(TestDocumentManagement, "uploaded_doc_id", ""), doc_ids)

    def test_04_delete_document(self):
        doc_id = getattr(TestDocumentManagement, "uploaded_doc_id", None)
        self.assertIsNotNone(doc_id, "Uploaded doc ID should exist.")

        print(f"\n--- Test 4: Deleting Document '{doc_id}' ---")
        response = self.client.delete(f"/api/documents/{doc_id}")
        print("Delete Response Code:", response.status_code)
        print("Delete Response Data:", response.json())

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["document_id"], doc_id)

        # Verify document is no longer listed
        list_resp = self.client.get("/api/documents")
        remaining_ids = [d["id"] for d in list_resp.json()["documents"]]
        self.assertNotIn(doc_id, remaining_ids)

    def test_05_delete_nonexistent_document(self):
        print("\n--- Test 5: Deleting Non-existent Document ---")
        response = self.client.delete("/api/documents/doc_nonexistent_999")
        self.assertEqual(response.status_code, 404)

if __name__ == "__main__":
    unittest.main()
