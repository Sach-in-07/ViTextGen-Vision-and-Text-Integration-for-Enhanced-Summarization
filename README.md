# 1. Web Scraping
The initial step involves extracting both textual content and images from web pages. We use BeautifulSoup for web scraping to extract the necessary data.

Text Extraction: BeautifulSoup navigates through HTML tags such as p, h1, and a to extract text. After the extraction, we clean the data by removing HTML tags and excess whitespace to ensure that the data is ready for further processing.

Image Extraction: For images, the process involves parsing the <img> tags to extract URLs. After verifying permissions and complying with website policies, the images are downloaded in various formats (JPEG, PNG, etc.) and stored locally.

# 2. Image Captioning
Once the images are extracted, we use the Vision Transformer (ViT) model to analyze and understand them. ViT divides images into patches and converts these patches into embeddings, which are high-dimensional representations of the visual features.

Encoder in ViT: The image patches, now converted into embeddings, are passed through multiple layers of the encoder, which uses self-attention and feed-forward networks. The self-attention mechanism helps the model focus on important regions of the image by computing relationships between the image patches.
Multi-Headed Attention
Multi-Headed Attention: This technique extends the self-attention model by allowing the model to jointly attend to information from different representation subspaces at different positions. The input is transformed into three vectors: Query (Q), Key (K), and Value (V). Each of these vectors is computed through distinct linear layers.

Dot Product of Query and Key: The Q and K vectors are multiplied using a dot product to create a scoring matrix that indicates the importance of each pixel in comparison to others. The resulting scores are crucial for determining how much attention each pixel should receive.

Scaling Down the Attention Scores: After generating the scores, they are scaled down by dividing by the square root of the dimension of the Key vectors. This scaling helps prevent large gradients during training, which could destabilize the learning process.

Softmax of the Scaled Scores: The softmax function is applied to the scaled scores to obtain attention weights, producing probability values between 0 and 1. Higher scores indicate more focus on the corresponding pixels. The softmax function helps in selecting which pixel to concentrate on for generating captions.

# 3. Text and Caption Integration
After generating captions for all images, the next step is to combine these captions with the original textual content extracted from the web page.

Data Structuring: The generated captions are placed at appropriate locations within the textual content to ensure a logical flow. This creates a cohesive structure where the captions complement the text.

Unified Dataset: The result is a comprehensive dataset that includes both the original web content and the generated image captions. This combined dataset is now ready for further processing, such as embedding generation and summarization.

# 4. Text Embedding
With the unified dataset ready, the next step is to convert the combined text (web content + image captions) into embeddings, which are numerical representations of the text.

Embedding Conversion: A Google Generative Model is used to convert the combined text and captions into high-dimensional embeddings. These embeddings capture the semantic meaning of the text, allowing for a more nuanced understanding of relationships between different parts of the content.

# 5. Storing Embeddings
Once the text is converted into embeddings, the next task is to store these high-dimensional vectors in a way that facilitates efficient retrieval and querying.

ChromaDB for Storage: The embeddings are stored in ChromaDB, which is optimized for handling high-dimensional vectors. This allows efficient storage and retrieval of embeddings.

Efficient Indexing and Querying: The database is used to index and retrieve embeddings based on similarity, allowing fast and efficient semantic search.

# 6. Summarization
The final step in the project involves generating concise summaries from the combined text embeddings using the FLAN-T5 model.

Input to FLAN-T5: The text embeddings generated in the previous step are fed into the FLAN-T5 model, a transformer-based model designed for text summarization.

FLAN-T5 Encoder: The encoder processes the embeddings using a multi-head self-attention mechanism, allowing the model to focus on the most relevant parts of the input.

FLAN-T5 Decoder: The decoder generates the summary word by word, ensuring coherence and relevance by attending to the encoded embeddings and previously generated words.

Attention Mechanism: Both self-attention and cross-attention mechanisms are employed in the encoder and decoder, enabling the model to focus on different parts of the input data while generating summaries that integrate information from text and images.
