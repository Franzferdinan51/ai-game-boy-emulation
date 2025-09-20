"""
Reinforcement Learning API Endpoints
"""
import os
import json
from flask import Blueprint, request, jsonify
from typing import Dict, Any, Optional

from ..rl.pyboy_rl_env import get_rl_environment, reset_rl_environment

# Create blueprint
rl_bp = Blueprint('rl', __name__, url_prefix='/api/rl')

@rl_bp.route('/initialize', methods=['POST'])
def initialize_rl_environment():
    """Initialize RL environment with ROM"""
    try:
        data = request.get_json()
        rom_path = data.get('rom_path')

        if not rom_path:
            return jsonify({'success': False, 'error': 'ROM path required'}), 400

        # Make ROM path absolute if it's relative
        if not os.path.isabs(rom_path):
            rom_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), rom_path)

        config = data.get('config', {})
        env = reset_rl_environment(rom_path, config)

        if env and env.initialized:
            return jsonify({
                'success': True,
                'message': 'RL environment initialized successfully',
                'rom_path': rom_path,
                'config': config
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to initialize RL environment'}), 500

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rl_bp.route('/step', methods=['POST'])
def rl_step():
    """Execute RL step"""
    try:
        data = request.get_json()
        action = data.get('action')

        if not action:
            return jsonify({'success': False, 'error': 'Action required'}), 400

        env = get_rl_environment()
        if not env or not env.initialized:
            return jsonify({'success': False, 'error': 'RL environment not initialized'}), 400

        result = env.step(action)
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rl_bp.route('/reset', methods=['POST'])
def rl_reset():
    """Reset RL environment"""
    try:
        env = get_rl_environment()
        if not env or not env.initialized:
            return jsonify({'success': False, 'error': 'RL environment not initialized'}), 400

        result = env.reset()
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rl_bp.route('/state', methods=['GET'])
def get_rl_state():
    """Get RL environment state"""
    try:
        env = get_rl_environment()
        if not env or not env.initialized:
            return jsonify({'success': False, 'error': 'RL environment not initialized'}), 400

        state_info = env.get_state_info()
        return jsonify({
            'success': True,
            'state': state_info,
            'screen_shape': env.get_screen().shape.tolist() if env.initialized else None
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rl_bp.route('/screen', methods=['GET'])
def get_rl_screen():
    """Get current screen observation"""
    try:
        env = get_rl_environment()
        if not env or not env.initialized:
            return jsonify({'success': False, 'error': 'RL environment not initialized'}), 400

        screen = env.get_screen()
        return jsonify({
            'success': True,
            'screen': screen.tolist(),
            'shape': screen.shape
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rl_bp.route('/memory/<int:address>', methods=['GET'])
def get_memory_value(address: int):
    """Get memory value at specific address"""
    try:
        env = get_rl_environment()
        if not env or not env.initialized:
            return jsonify({'success': False, 'error': 'RL environment not initialized'}), 400

        value = env.get_memory_value(address)
        return jsonify({
            'success': True,
            'address': address,
            'value': value,
            'hex': hex(value)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rl_bp.route('/rules/reward', methods=['POST'])
def add_reward_rule():
    """Add a reward rule"""
    try:
        data = request.get_json()
        address = data.get('address')
        operator = data.get('operator')
        reward = data.get('reward')
        label = data.get('label')

        if None in [address, operator, reward, label]:
            return jsonify({'success': False, 'error': 'All fields required'}), 400

        env = get_rl_environment()
        if not env or not env.initialized:
            return jsonify({'success': False, 'error': 'RL environment not initialized'}), 400

        env.add_reward_rule(address, operator, reward, label)

        return jsonify({
            'success': True,
            'message': 'Reward rule added successfully',
            'rule': {
                'address': address,
                'operator': operator,
                'reward': reward,
                'label': label
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rl_bp.route('/rules/done', methods=['POST'])
def add_done_rule():
    """Add a done rule"""
    try:
        data = request.get_json()
        address = data.get('address')
        operator = data.get('operator')
        label = data.get('label')

        if None in [address, operator, label]:
            return jsonify({'success': False, 'error': 'All fields required'}), 400

        env = get_rl_environment()
        if not env or not env.initialized:
            return jsonify({'success': False, 'error': 'RL environment not initialized'}), 400

        env.add_done_rule(address, operator, label)

        return jsonify({
            'success': True,
            'message': 'Done rule added successfully',
            'rule': {
                'address': address,
                'operator': operator,
                'label': label
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rl_bp.route('/close', methods=['POST'])
def close_rl_environment():
    """Close RL environment"""
    try:
        env = get_rl_environment()
        if env:
            env.close()

        return jsonify({'success': True, 'message': 'RL environment closed'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rl_bp.route('/info', methods=['GET'])
def get_rl_info():
    """Get RL system information"""
    try:
        env = get_rl_environment()
        info = {
            'initialized': env is not None and env.initialized,
            'rom_path': env.rom_path if env else None,
            'reward_rules_count': len(env.reward_rules) if env else 0,
            'done_rules_count': len(env.done_rules) if env else 0,
            'available_actions': [
                'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'SELECT', 'START',
                'RELEASE_UP', 'RELEASE_DOWN', 'RELEASE_LEFT', 'RELEASE_RIGHT',
                'RELEASE_A', 'RELEASE_B', 'RELEASE_SELECT', 'RELEASE_START', 'PASS'
            ]
        }

        if env and env.initialized:
            info['state'] = env.get_state_info()

        return jsonify({
            'success': True,
            'info': info
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500