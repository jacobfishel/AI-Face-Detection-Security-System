# Creating a server using python that involves threading

#Steps for creating a server:
#1. Socket
#2. Bind the socket
#3. Listen
#4. accept

import socket
import threading
from backend.functions import handle_connection

# Define the server address and port
HOST = '127.0.0.1'
PORT = 12345

# Create the socket object
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# Bind the socket to the address and port
server.bind((HOST, PORT))

# Listen for incoming connections
server.listen()
print(f"Server is listening on {HOST}:{PORT}")

try:
  while True:
    # Accept a client connection
    client_socket, client_address = server.accept()
    print(f"Accepted a client on {HOST}:{PORT}")
    # Create a new thread for each client connection
    client_thread = threading.Thread(target=handle_connection, args=(client_socket, client_address))
    client_thread.start()

except KeyboardInterrupt:
  print("\n[!] KeyboardInterupt recieved. Closing server...")
  
finally:
  server.close()
  print("Server closed cleanly")




