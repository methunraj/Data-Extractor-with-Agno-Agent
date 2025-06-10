import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { extractedData, fileName, llmProvider, model, apiKey, temperature } = body;

    // Validate required fields
    // For googleAI provider, API key might be optional (using default key)
    if (!extractedData || !fileName || !model) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields: extractedData, fileName, or model' },
        { status: 400 }
      );
    }
    
    // Use API key from request or get from environment for googleAI
    const finalApiKey = apiKey || (llmProvider === 'googleAI' ? process.env.GOOGLE_API_KEY : null);
    
    if (!finalApiKey) {
      return NextResponse.json(
        { success: false, error: 'API key is required' },
        { status: 400 }
      );
    }

    // Forward request to Python backend
    const pythonBackendUrl = process.env.AGNO_BACKEND_URL || 'http://localhost:8001';
    
    const response = await fetch(`${pythonBackendUrl}/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        json_data: extractedData,  // Changed from extracted_data to match Python API
        file_name: fileName,
        description: `Extracted from ${fileName} using ${llmProvider}/${model}`,
        api_key: finalApiKey,  // Use the resolved API key
        model: model  // Python backend will use this model for Agno processing
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Python backend error:', errorText);
      return NextResponse.json(
        { success: false, error: 'Agno processing failed' },
        { status: 500 }
      );
    }

    const result = await response.json();
    
    // If successful, add the backend URL to the download URL for frontend access
    if (result.success && result.download_url) {
      // The download URL is already properly formatted as /download/{file_id}
      result.download_url = `${pythonBackendUrl}${result.download_url}`;
      
      // Add token usage information if needed (Python backend doesn't currently provide this)
      // You can add token tracking in the Python backend if needed
      result.token_usage = {
        input_tokens: 0,  // Placeholder - implement in Python if needed
        output_tokens: 0,
        total_tokens: 0
      };
      
      // Add processing cost if needed
      result.processingCost = 0;  // Placeholder - calculate based on actual usage
    }
    
    return NextResponse.json(result);

  } catch (error) {
    console.error('Agno API route error:', error);
    return NextResponse.json(
      { success: false, error: 'Internal server error' },
      { status: 500 }
    );
  }
}