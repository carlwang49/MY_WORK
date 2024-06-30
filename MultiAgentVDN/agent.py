import torch
import torch.nn.functional as F
from EVBuildingEnv import EVBuildingEnv

    
def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10, chunk_size=10, grad_clip_norm=5):
    _chunk_size = chunk_size if q.recurrent else 1
    for _ in range(update_iter):
        s, a, r, s_prime, connected = memory.sample_chunk(batch_size, _chunk_size)

        hidden = q.init_hidden(batch_size)
        target_hidden = q_target.init_hidden(batch_size)
        loss = 0
        for step_i in range(_chunk_size):
            q_out, hidden = q(s[:, step_i, :, :], hidden)
            q_a = q_out.gather(2, a[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)
            sum_q = q_a.sum(dim=1, keepdims=True)

            max_q_prime, target_hidden = q_target(s_prime[:, step_i, :, :], target_hidden.detach())
            max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1)
            target_q = r[:, step_i, :].sum(dim=1, keepdims=True)
            target_q += gamma * max_q_prime.sum(dim=1, keepdims=True) * (1 - connected[:, step_i])

            loss += F.smooth_l1_loss(sum_q, target_q.detach())
            
            connected_mask = connected[:, step_i].squeeze(-1).bool()
            hidden[connected_mask] = q.init_hidden(len(hidden[connected_mask]))
            target_hidden[connected_mask] = q_target.init_hidden(len(target_hidden[connected_mask]))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), grad_clip_norm, norm_type=2)
        optimizer.step()
        
        
def test(env: EVBuildingEnv, num_episodes, q, ev_request_dict, ev_departure_dict):
    score = 0
    for episode_i in range(num_episodes):
        
        step_counter = 0
        state = env.reset()
        env.timestamp = env.start_time
        
        with torch.no_grad():
            hidden = q.init_hidden()
            while env.timestamp <= env.end_time: 
                
                # add EVs to the environment, if there are EVs that have arrived at the current time
                current_requests = ev_request_dict.get(env.timestamp, []) # get the EVs that have arrived at the current time
                if current_requests:
                    for ev in current_requests:
                        env.add_ev(ev['requestID'], 
                                ev['arrival_time'], 
                                ev['departure_time'], 
                                ev['initial_soc'], 
                                ev['departure_soc'])
                        
                        env.current_parking_number += 1 # increase the number of EVs in the environment
                            
                # Remove EVs that departed at the current time
                current_departures = ev_departure_dict.get(env.timestamp, [])
                if current_departures:
                    for ev in current_departures:
                        request_id = ev['requestID']
                        for agent_id, data in env.ev_data.items():
                            if data['requestID'] == request_id:
                                env.remove_ev(agent_id)
                                env.current_parking_number -= 1
                                break 
                        
                step_counter += 1
                action, hidden = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon=0)
                action = action[0].data.cpu().numpy().tolist()
                action = [env.action_values[int(a)] for a in action]
                next_state, reward, connected, info = env.step(action, env.timestamp)
                score += sum(reward)
                state = next_state
                
                if env.timestamp >= env.end_time: 
                    break
                
    return score / num_episodes