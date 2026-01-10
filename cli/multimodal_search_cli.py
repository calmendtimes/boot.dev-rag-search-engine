import argparse
import mimetypes
import lib.semantic_search as SS
import lib.multimodal_search as MMS


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")    
    verify_parser = subparsers.add_parser("verify_image_embedding", help="image embedding") 
    verify_parser.add_argument("path", type=str, help="Image path")
    image_search_parser = subparsers.add_parser("image_search", help="Search by image") 
    image_search_parser.add_argument("path", type=str, help="Image path")
    
    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":       
            documents = SS.load_movies()
            mms = MMS.MultimodalSearch(documents)
            embedding = mms.embed_image(args.path)
            print(f"Embedding shape: {embedding.shape[0]} dimensions")
        case "image_search":
            documents = SS.load_movies()
            mms = MMS.MultimodalSearch(documents)
            result = mms.search_with_image(args.path)
            for i in range(len(result)):
                print(f"{i+1}. {result[i][1]['title']} (similarity: {result[i][0]:.3f})")
                print(f"     {result[i][1]['description'][:100]}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()