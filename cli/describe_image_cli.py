import argparse
import mimetypes
import lib.gemini as gemini


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    parser.add_argument("--image", type=str, help="Image")
    parser.add_argument("--query", type=str, help="Query")
    
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, 'rb') as f: 
        image = f.read()

    prompt = "Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:\n" +\
                "- Synthesize visual and textual information\n" +\
                "- Focus on movie-specific details (actors, scenes, style, etc.)\n" +\
                "- Return only the rewritten query, without any additional commentary"
    
    response = gemini.request_with_image(prompt, image, mime, args.query)
    
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")



if __name__ == "__main__":
    main()