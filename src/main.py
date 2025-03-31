from .api import run_api
import argparse

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Memory Orb System')
    parser.add_argument('--port', type=int, default=5000, help='API server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='API server host')  # Changed to 0.0.0.0 to allow access from any address
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"Starting API server at {args.host}:{args.port}...")
    run_api(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()