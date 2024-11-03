from django.shortcuts import render
from vector import Vectorizer
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
import os

vectorizer = Vectorizer()

@csrf_exempt
def home(request):
    """
    Handle the home page view for processing uploaded files and returning similar articles.

    This view handles both GET and POST requests. For POST requests, it processes the uploaded file,
    extracts similar articles based on the provided query, and returns the results. For GET requests,
    it simply renders the home page.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object with the rendered template.
    """
    if request.method == 'POST':
        query = request.POST['query']
        top_result = int(request.POST['top_result'])
        file = request.FILES['file']

        # Save the uploaded file
        file_path = default_storage.save(file.name, ContentFile(file.read()))

        try:
            # Process the file and get similar articles
            similarity_scores, docs, keywords = vectorizer.get_similar_articles(query, top_result, file_path)

            # Truncate docs and keywords to match the length of similarity_scores
            docs = docs[:len(similarity_scores)]
            keywords = keywords[:len(similarity_scores)]

            # Extract only the first paragraph from each document
            first_paragraphs = [doc.split('\n\n')[0] for doc in docs]

            results = []
            for i, (doc, keyword) in enumerate(zip(first_paragraphs, keywords)):
                results.append({
                    'similarity_score': similarity_scores[i],
                    'keywords': keyword,
                    'document': doc
                })

            # Sort results by similarity score in descending order
            results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)

            return render(request, 'Chatbot/result.html', {'results': results})
        finally:
            # Delete the uploaded file after processing
            if default_storage.exists(file_path):
                default_storage.delete(file_path)

    return render(request, 'Chatbot/home.html')