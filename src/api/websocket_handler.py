'''
Central hub for all messages from backend to frontend and vice versa.
'''

async def handle_message(websocket):
    ''''
        recieve messages from frontend
        parse JSON 'action' field
        call corresponding function in backend
        send status updates / final prediction
    '''


async def send_update(websocket, message_type: str, data):
    '''
        format JSON message and send over websocket
    '''
    await websocket.send_text(message_type)


