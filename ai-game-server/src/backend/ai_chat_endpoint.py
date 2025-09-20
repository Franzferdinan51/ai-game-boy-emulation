@app.route('/api/ai-chat', methods=['POST'])
def ai_chat():
    """Send a message to the AI and get a response"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400
        
        data = request.json
        user_message = data.get('message', '')
        action_history_data = data.get('action_history', [])
        current_goal = data.get('current_goal', '')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Get the default AI API (or use the first available)
        api_name = "gemini"  # Default for now
        if api_name not in ai_apis:
            # Try to find any available AI API
            available_apis = list(ai_apis.keys())
            if not available_apis:
                return jsonify({"error": "No AI APIs available"}), 500
            api_name = available_apis[0]
        
        # Get screen from active emulator for context
        emulator = emulators[game_state["active_emulator"]]
        screen_array = emulator.get_screen()
        
        # Convert screen to bytes for AI API
        img_buffer = io.BytesIO()
        Image.fromarray(screen_array).save(img_buffer, format='JPEG')
        img_bytes = img_buffer.getvalue()
        
        # Prepare context for the AI
        context = {
            "current_goal": current_goal,
            "action_history": action_history_data,
            "game_type": game_state["active_emulator"].upper()
        }
        
        # Get response from AI
        ai_connector = ai_apis[api_name]
        response_text = ai_connector.chat_with_ai(user_message, img_bytes, context)
        
        logger.info(f"AI chat message from user: {user_message}")
        return jsonify({
            "response": response_text,
            "api_used": api_name
        }), 200
        
    except Exception as e:
        logger.error(f"Error in AI chat: {e}")
        return jsonify({"error": "Internal server error"}), 500