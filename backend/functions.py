import socket

def handle_connection(client_socket, client_address):
  print("Starging the handle connection")
  # read the incoming data
  #write it back
  with client_socket:
    print(f"Connected by {client_address}")
    while True:
      # Recieve data from the client
      data = client_socket.recv(1024)
      if not data:
        break
      # Echo the recieverd data back to the client
      client_socket.sendall(data)
      text = data.decode().strip()
      print(text)
    