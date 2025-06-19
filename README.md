# CLIP Image Search API

## Description
A Flask API powered by OpenAI's CLIP model for image feature extraction and similarity calculation.

## Features
- Extract 512-dimensional feature vectors from images
- Calculate cosine similarity between image features
- RESTful API endpoints
- Built-in Gradio interface for testing

## API Endpoints

### Health Check
```
GET /health
```

### Extract Image Features
```
POST /extract
Content-Type: application/json

{
  "image": "base64_encoded_image_string"
}
```

Response:
```json
{
  "success": true,
  "features": [0.1, -0.2, 0.3, ...],
  "feature_size": 512
}
```

### Calculate Similarity
```
POST /similarity
Content-Type: application/json

{
  "features1": [0.1, -0.2, 0.3, ...],
  "features2": [0.2, -0.1, 0.4, ...]
}
```

Response:
```json
{
  "success": true,
  "similarity": 0.8534
}
```

## Usage from ASP.NET Core

```csharp
public async Task<float[]> ExtractImageFeatures(byte[] imageBytes)
{
    using var client = new HttpClient();
    var base64Image = Convert.ToBase64String(imageBytes);
    var payload = new { image = base64Image };
    
    var response = await client.PostAsync(
        "https://your-space-name-your-username.hf.space/extract",
        new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json")
    );
    
    var result = await response.Content.ReadAsStringAsync();
    var parsed = JsonSerializer.Deserialize<ExtractResponse>(result);
    
    return parsed.features;
}
```

## Deployment Instructions

1. Create a new Space on Hugging Face
2. Choose "Gradio" as the Space SDK
3. Upload `app.py` and `requirements.txt`
4. Your API will be available at: `https://your-space-name-your-username.hf.space`

## Notes
- The model is automatically downloaded during the first build
- Supports both CPU and GPU inference
- CORS enabled for cross-origin requests