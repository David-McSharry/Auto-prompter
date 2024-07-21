import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from roles import PromptEngineer
from datasets import load_dataset
from utils import make_PE_feedback, test_prompt_on_benchmark_async, load_questions
import json

app = FastAPI()

# Add CORS middleware to allow cross-origin requests (important for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configuration
config = {
    "prompt_number": 2,
    "num_wrong_feedback_questions": 1,
    "num_benchmark_samples": 2,
}

# Load the dataset
dataset = load_questions()

# Initialize PromptEngineer

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            prompt_engineer = PromptEngineer(prompt_number=config["prompt_number"])

            # Wait for a message from the client
            message = await websocket.receive_text()
            
            # Check if the message is to start the main loop
            if message == "start_main_loop":
                # Run the main loop
                for i in range(config["prompt_number"]):
                    print('===================================')
                    # Generate next prompt
                    next_prompt = prompt_engineer.generate_next_prompt()
                    
                    # Update and broadcast system state
                    await manager.broadcast(json.dumps(prompt_engineer.system_state))
                    
                    # Test the prompt
                    score, correct, wrong, invalid = await test_prompt_on_benchmark_async(
                        next_prompt,
                        dataset,
                        num_samples=config["num_benchmark_samples"],
                        dataset_type='logicqa2.0'
                    )
                    
                    # Generate feedback
                    feedback = make_PE_feedback(
                        score,
                        wrong,
                        num_wrongly_answered=config["num_wrong_feedback_questions"],
                        invalid_answer_decimals=invalid
                    )
                    
                    # Add feedback to PromptEngineer
                    prompt_engineer.add_user_feedback_response(
                        wrongly_answered_qs=wrong,
                        correctly_answered_qs=correct,
                        user_feedback=feedback,
                        score=score
                    )
                    
                    # Update and broadcast system state again
                    await manager.broadcast(json.dumps(prompt_engineer.system_state))
                    
                    # Small delay to prevent flooding
                    await asyncio.sleep(0.1)
                
                # Notify the client that the loop is complete
                await websocket.send_text("main_loop_complete")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
