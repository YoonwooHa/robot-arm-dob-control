import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
import time
from tqdm import tqdm

# 한글 폰트 설정 (옵션)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# matplotlib backend 설정 (Windows에서 그래프 창 유지)
import matplotlib
matplotlib.use('TkAgg')  # Windows에서 안정적인 backend

# Interactive mode 활성화
plt.ion()

class RobotManipulator:
    """2자유도 로봇 매니퓰레이터 동역학 모델"""
    def __init__(self):
        # 시스템 파라미터 (논문 값)
        self.p1 = 3.473  # kg⋅m²
        self.p2 = 0.196  # kg⋅m²
        self.p3 = 0.242  # kg⋅m²
        self.fd1 = 5.3   # N⋅m⋅s
        self.fd2 = 1.1   # N⋅m⋅s
        
    def mass_matrix(self, q):
        """관성 행렬 M(q)"""
        q1, q2 = q[0], q[1]
        M11 = self.p1 + 2*self.p3*np.cos(q2)
        M12 = self.p2 + self.p3*np.cos(q2)
        M21 = self.p2 + self.p3*np.cos(q2)
        M22 = self.p2
        return np.array([[M11, M12], [M21, M22]])
    
    def coriolis_matrix(self, q, q_dot):
        """원심력-코리올리 행렬 C(q,q̇)"""
        q1, q2 = q[0], q[1]
        q1_dot, q2_dot = q_dot[0], q_dot[1]
        C11 = -self.p3*np.sin(q2)*q2_dot
        C12 = -self.p3*np.sin(q2)*(q1_dot + q2_dot)
        C21 = self.p3*np.sin(q2)*q1_dot
        C22 = 0
        return np.array([[C11, C12], [C21, C22]])
    
    def friction_matrix(self):
        """마찰 행렬 F"""
        return np.array([[self.fd1, 0], [0, self.fd2]])
    
    def external_disturbance(self, t):
        """외부 외란 - 더 큰 크기와 복잡한 패턴"""
        # 기본 사인파 외란 (크기 증가)
        sd1_base = 8*np.sin(t)  # 3 → 8로 증가
        sd2_base = 1.5*np.sin(t)  # 0.2 → 1.5로 증가
        
        # 스텝 외란 (주기적)
        step_disturbance1 = 2 if (int(t) % 4 < 2) else -2
        step_disturbance2 = 1 if (int(t) % 6 < 3) else -1
        
        # 시간에 따라 증가하는 외란
        time_varying1 = 0.5 * t * np.sin(t) / (1 + 0.1*t)  # 시간에 따라 변화
        time_varying2 = 0.3 * t * np.cos(t) / (1 + 0.1*t)
        
        # 전체 외란 조합
        sd1 = sd1_base + step_disturbance1 + time_varying1
        sd2 = sd2_base + step_disturbance2 + time_varying2
        
        return np.array([sd1, sd2])
    
    def dynamics(self, t, state, tau):
        """로봇 동역학"""
        q = state[:2]
        q_dot = state[2:]
        
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        F = self.friction_matrix()
        sd = self.external_disturbance(t)
        
        # M(q)q̈ + C(q,q̇)q̇ + F(q̇) + sd = τ
        # q̈ = M^(-1)[τ - C(q,q̇)q̇ - F(q̇) - sd]
        M_inv = np.linalg.inv(M)
        q_ddot = M_inv @ (tau - C @ q_dot - F @ q_dot - sd)
        
        return np.concatenate([q_dot, q_ddot])

class NeuralNetwork(nn.Module):
    """신경망 기본 클래스"""
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ActorNetwork(NeuralNetwork):
    """액터 네트워크 - 미지 동역학 f(x) 추정"""
    def __init__(self, state_dim=4, hidden_size=10, output_dim=2):
        super(ActorNetwork, self).__init__(state_dim, hidden_size, output_dim)

class CriticNetwork(NeuralNetwork):
    """크리틱 네트워크 - 가치 함수 V(x) 추정"""
    def __init__(self, state_dim=2, hidden_size=10, output_dim=1):
        super(CriticNetwork, self).__init__(state_dim, hidden_size, output_dim)

class DisturbanceObserver:
    """적응형 신경 외란 관측기"""
    def __init__(self, alpha=20):
        self.alpha = alpha
        self.h = np.zeros(2)  # 내부 상태
        
    def update(self, x, u, f_hat, dt):
        """외란 관측기 업데이트"""
        x_n = x[2:]  # 속도 상태
        # ḣ = -α(h + αx_n) - α(Ŵ_a^T φ_a + g(x)u)
        h_dot = -self.alpha * (self.h + self.alpha * x_n) - self.alpha * (f_hat + u)
        self.h += h_dot * dt
        
        # 외란 추정값
        d_hat = self.h + self.alpha * x_n
        return d_hat

class DisturbanceObserverActorCriticController:
    """외란 관측기 기반 액터-크리틱 제어기"""
    def __init__(self):
        self.robot = RobotManipulator()
        
        # 신경망 초기화
        self.actor = ActorNetwork()
        self.critic = CriticNetwork()
        self.disturbance_observer = DisturbanceObserver()
        
        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.01)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.02)
        
        # 제어 파라미터 (논문 값)
        self.k1 = 30
        self.k2 = 10
        self.kn = 10
        self.beta = 100
        
        # 학습 파라미터
        self.ka1 = 20
        self.ka2 = 1
        self.ka3 = 5
        self.kc1 = 2
        self.kc2 = 0.1
        self.gamma = 0.1  # 할인 인자
        
        # 성능 지수 가중치
        self.Q = np.diag([50, 200])
        self.R = np.diag([0.1, 0.1])
        
        # 저장용 변수
        self.actor_losses = []
        self.critic_losses = []
        
    def reference_trajectory(self, t):
        """참조 궤적"""
        yr1 = 0.6 * np.sin(3.14 * t) * (1 - np.exp(-t))
        yr2 = 0.8 * np.sin(3.14 * t) * (1 - np.exp(-t))
        
        # 1차, 2차 미분
        yr1_dot = 0.6 * (3.14 * np.cos(3.14 * t) * (1 - np.exp(-t)) + np.sin(3.14 * t) * np.exp(-t))
        yr2_dot = 0.8 * (3.14 * np.cos(3.14 * t) * (1 - np.exp(-t)) + np.sin(3.14 * t) * np.exp(-t))
        
        yr1_ddot = 0.6 * (-3.14**2 * np.sin(3.14 * t) * (1 - np.exp(-t)) + 
                          2 * 3.14 * np.cos(3.14 * t) * np.exp(-t) - np.sin(3.14 * t) * np.exp(-t))
        yr2_ddot = 0.8 * (-3.14**2 * np.sin(3.14 * t) * (1 - np.exp(-t)) + 
                          2 * 3.14 * np.cos(3.14 * t) * np.exp(-t) - np.sin(3.14 * t) * np.exp(-t))
        
        return np.array([yr1, yr2]), np.array([yr1_dot, yr2_dot]), np.array([yr1_ddot, yr2_ddot])
    
    def compute_filtered_errors(self, state, yr, yr_dot, yr_ddot):
        """필터링된 추적 오차 계산"""
        q = state[:2]
        q_dot = state[2:]
        
        z1 = q - yr
        z2 = q_dot - yr_dot + self.k1 * z1
        
        return z1, z2
    
    def compute_cost(self, z1, u):
        """즉시 비용 함수"""
        return z1.T @ self.Q @ z1 + u.T @ self.R @ u
    
    def update_networks(self, state, z1, z2, u, reward, next_state, dt):
        """액터-크리틱 네트워크 업데이트"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        z1_tensor = torch.FloatTensor(z1).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        next_z1_tensor = torch.FloatTensor(next_state[:2] - self.reference_trajectory(0)[0]).unsqueeze(0)
        
        # 크리틱 네트워크 업데이트
        current_value = self.critic(z1_tensor)
        next_value = self.critic(next_z1_tensor)
        target_value = reward + self.gamma * next_value
        
        critic_loss = nn.MSELoss()(current_value, target_value.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 액터 네트워크 업데이트
        f_hat = self.actor(state_tensor)
        
        # 복합 적응 법칙 (예측 오차 + 모델링 오차)
        prediction_error = current_value.item()
        modeling_error = np.linalg.norm(z2)  # 모델링 오차 근사
        
        actor_loss = prediction_error + 0.1 * modeling_error**2
        actor_loss_tensor = torch.tensor(actor_loss, requires_grad=True)
        
        self.actor_optimizer.zero_grad()
        actor_loss_tensor.backward()
        self.actor_optimizer.step()
        
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss.item())
        
        return f_hat.detach().numpy().flatten()
    
    def control_law(self, t, state, dt):
        """제어 법칙"""
        q = state[:2]
        q_dot = state[2:]
        
        # 참조 궤적
        yr, yr_dot, yr_ddot = self.reference_trajectory(t)
        
        # 필터링된 오차
        z1, z2 = self.compute_filtered_errors(state, yr, yr_dot, yr_ddot)
        
        # 액터 네트워크로 미지 동역학 추정
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            f_hat = self.actor(state_tensor).numpy().flatten()
        
        # 외란 관측기 업데이트
        M_inv = np.linalg.inv(self.robot.mass_matrix(q))
        d_hat = self.disturbance_observer.update(state, np.zeros(2), f_hat, dt)
        
        # 제어 입력 계산
        # u = M(q)[yr_ddot - k*z - kn*zn - f_hat - d_hat]
        k_z = self.k1 * z1 + self.k2 * z2
        u = self.robot.mass_matrix(q) @ (yr_ddot - k_z - self.kn * z2 - f_hat - d_hat)
        
        # 제어 입력 제한
        u = np.clip(u, -50, 50)
        
        # 네트워크 업데이트 (강화학습)
        reward = -self.compute_cost(z1, u)
        next_state = state.copy()  # 간단한 근사
        f_hat_updated = self.update_networks(state, z1, z2, u, reward, next_state, dt)
        
        return u, z1, z2, f_hat_updated, d_hat

class RealTimeVisualizer:
    """실시간 시각화 클래스"""
    def __init__(self, t_span, dt):
        self.t_span = t_span
        self.dt = dt
        self.t_eval = np.arange(0, t_span[1], dt)
        
        # 그래프 설정 - 3x3으로 확장하여 외란 그래프 추가
        self.fig, self.axes = plt.subplots(3, 3, figsize=(18, 12))
        self.fig.suptitle('Disturbance Observer-based Actor-Critic Control - Real-time Simulation', fontsize=14)
        
        # 데이터 저장용
        self.time_data = []
        self.q1_data = []
        self.q2_data = []
        self.q1_ref_data = []
        self.q2_ref_data = []
        self.error1_data = []
        self.error2_data = []
        self.u1_data = []
        self.u2_data = []
        self.actor_loss_data = []
        self.critic_loss_data = []
        self.disturbance1_data = []
        self.disturbance2_data = []
        self.d_hat1_data = []
        self.d_hat2_data = []
        
        # 그래프 라인 초기화
        self.setup_plots()
        
        # 진행 상황 표시
        self.progress_bar = tqdm(total=len(self.t_eval), desc="Simulation Progress")
        
    def setup_plots(self):
        """그래프 설정"""
        # Joint 1 추적 성능
        self.axes[0, 0].set_title('Joint 1 Tracking Performance')
        self.axes[0, 0].set_xlabel('Time (s)')
        self.axes[0, 0].set_ylabel('Angle (rad)')
        self.line_q1, = self.axes[0, 0].plot([], [], 'b-', label='Actual', linewidth=2)
        self.line_q1_ref, = self.axes[0, 0].plot([], [], 'r--', label='Reference', linewidth=2)
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True)
        
        # Joint 2 추적 성능
        self.axes[1, 0].set_title('Joint 2 Tracking Performance')
        self.axes[1, 0].set_xlabel('Time (s)')
        self.axes[1, 0].set_ylabel('Angle (rad)')
        self.line_q2, = self.axes[1, 0].plot([], [], 'b-', label='Actual', linewidth=2)
        self.line_q2_ref, = self.axes[1, 0].plot([], [], 'r--', label='Reference', linewidth=2)
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True)
        
        # 추적 오차
        self.axes[0, 1].set_title('Joint 1 Tracking Error')
        self.axes[0, 1].set_xlabel('Time (s)')
        self.axes[0, 1].set_ylabel('Error (rad)')
        self.line_error1, = self.axes[0, 1].plot([], [], 'g-', linewidth=2)
        self.axes[0, 1].grid(True)
        
        self.axes[1, 1].set_title('Joint 2 Tracking Error')
        self.axes[1, 1].set_xlabel('Time (s)')
        self.axes[1, 1].set_ylabel('Error (rad)')
        self.line_error2, = self.axes[1, 1].plot([], [], 'g-', linewidth=2)
        self.axes[1, 1].grid(True)
        
        # 제어 입력
        self.axes[0, 2].set_title('Joint 1 Control Input')
        self.axes[0, 2].set_xlabel('Time (s)')
        self.axes[0, 2].set_ylabel('Torque (N⋅m)')
        self.line_u1, = self.axes[0, 2].plot([], [], 'm-', linewidth=2)
        self.axes[0, 2].grid(True)
        
        self.axes[1, 2].set_title('Joint 2 Control Input')
        self.axes[1, 2].set_xlabel('Time (s)')
        self.axes[1, 2].set_ylabel('Torque (N⋅m)')
        self.line_u2, = self.axes[1, 2].plot([], [], 'm-', linewidth=2)
        self.axes[1, 2].grid(True)
        
        # 외란 및 관측값 (새로 추가)
        self.axes[2, 0].set_title('Joint 1 Disturbance')
        self.axes[2, 0].set_xlabel('Time (s)')
        self.axes[2, 0].set_ylabel('Disturbance (N⋅m)')
        self.line_d1, = self.axes[2, 0].plot([], [], 'r-', label='Actual', linewidth=2)
        self.line_d1_hat, = self.axes[2, 0].plot([], [], 'b--', label='Estimated', linewidth=2)
        self.axes[2, 0].legend()
        self.axes[2, 0].grid(True)
        
        self.axes[2, 1].set_title('Joint 2 Disturbance')
        self.axes[2, 1].set_xlabel('Time (s)')
        self.axes[2, 1].set_ylabel('Disturbance (N⋅m)')
        self.line_d2, = self.axes[2, 1].plot([], [], 'r-', label='Actual', linewidth=2)
        self.line_d2_hat, = self.axes[2, 1].plot([], [], 'b--', label='Estimated', linewidth=2)
        self.axes[2, 1].legend()
        self.axes[2, 1].grid(True)
        
        # 학습 곡선 (실시간)
        self.axes[2, 2].set_title('Learning Progress')
        self.axes[2, 2].set_xlabel('Steps')
        self.axes[2, 2].set_ylabel('Loss')
        self.line_actor_loss, = self.axes[2, 2].plot([], [], 'b-', label='Actor Loss', linewidth=2)
        self.line_critic_loss, = self.axes[2, 2].plot([], [], 'r-', label='Critic Loss', linewidth=2)
        self.axes[2, 2].legend()
        self.axes[2, 2].grid(True)
        self.axes[2, 2].set_yscale('symlog')  # 로그 스케일로 손실 변화 보기
        
        # 축 범위 설정
        self.axes[0, 0].set_ylim(-1.0, 1.0)
        self.axes[1, 0].set_ylim(-1.5, 1.5)
        self.axes[0, 1].set_ylim(-1.0, 1.0)
        self.axes[1, 1].set_ylim(-1.0, 1.0)
        self.axes[0, 2].set_ylim(-100, 100)
        self.axes[1, 2].set_ylim(-100, 100)
        self.axes[2, 0].set_ylim(-20, 20)
        self.axes[2, 1].set_ylim(-10, 10)
        
        for ax in self.axes.flat:
            ax.set_xlim(0, self.t_span[1])
    
    def update_plots(self, t, state, u, error, ref, actor_loss, critic_loss, disturbance, d_hat):
        """그래프 업데이트 - 외란 데이터 추가"""
        # 데이터 추가
        self.time_data.append(t)
        self.q1_data.append(state[0])
        self.q2_data.append(state[1])
        self.q1_ref_data.append(ref[0])
        self.q2_ref_data.append(ref[1])
        self.error1_data.append(error[0])
        self.error2_data.append(error[1])
        self.u1_data.append(u[0])
        self.u2_data.append(u[1])
        self.actor_loss_data.append(actor_loss)
        self.critic_loss_data.append(critic_loss)
        self.disturbance1_data.append(disturbance[0])
        self.disturbance2_data.append(disturbance[1])
        self.d_hat1_data.append(d_hat[0])
        self.d_hat2_data.append(d_hat[1])
        
        # 라인 업데이트
        self.line_q1.set_data(self.time_data, self.q1_data)
        self.line_q1_ref.set_data(self.time_data, self.q1_ref_data)
        self.line_q2.set_data(self.time_data, self.q2_data)
        self.line_q2_ref.set_data(self.time_data, self.q2_ref_data)
        self.line_error1.set_data(self.time_data, self.error1_data)
        self.line_error2.set_data(self.time_data, self.error2_data)
        self.line_u1.set_data(self.time_data, self.u1_data)
        self.line_u2.set_data(self.time_data, self.u2_data)
        
        # 외란 데이터 업데이트
        self.line_d1.set_data(self.time_data, self.disturbance1_data)
        self.line_d1_hat.set_data(self.time_data, self.d_hat1_data)
        self.line_d2.set_data(self.time_data, self.disturbance2_data)
        self.line_d2_hat.set_data(self.time_data, self.d_hat2_data)
        
        # 학습 곡선 업데이트
        steps = list(range(len(self.actor_loss_data)))
        self.line_actor_loss.set_data(steps, self.actor_loss_data)
        self.line_critic_loss.set_data(steps, self.critic_loss_data)
        
        # 그래프 다시 그리기 (더 빠른 업데이트를 위해)
        if len(self.time_data) % 10 == 0:  # 10스텝마다 업데이트
            # 메인 figure 업데이트
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        
        # 진행 상황 업데이트
        self.progress_bar.update(1)
        self.progress_bar.set_postfix({
            'E1': f'{error[0]:.4f}',
            'E2': f'{error[1]:.4f}',
            'D1': f'{disturbance[0]:.2f}',
            'D2': f'{disturbance[1]:.2f}',
            'A_Loss': f'{actor_loss:.3f}',
            'C_Loss': f'{critic_loss:.3f}'
        })
    
    def show_learning_curves(self):
        """학습 곡선 표시"""
        self.progress_bar.close()
        
        # 새로운 창에서 학습 곡선 표시
        fig_learning, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.actor_loss_data)
        ax1.set_title('Actor Network Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        ax2.plot(self.critic_loss_data)
        ax2.set_title('Critic Network Loss')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 성능 지표 출력
        print("\n=== Final Performance Metrics ===")
        if len(self.error1_data) > 1000:
            max_error_1 = np.max(np.abs(self.error1_data[-1000:]))
            max_error_2 = np.max(np.abs(self.error2_data[-1000:]))
            print(f"Joint 1 Max Steady-State Error: {max_error_1:.6f} rad")
            print(f"Joint 2 Max Steady-State Error: {max_error_2:.6f} rad")
        
        if len(self.u1_data) > 0:
            avg_control = np.mean([np.abs(self.u1_data), np.abs(self.u2_data)])
            print(f"Average Control Input Magnitude: {avg_control:.4f} N⋅m")

class RobotArm2DVisualizer:
    """2D 로봇 팔 시각화"""
    def __init__(self, L1=1.0, L2=0.8):
        self.L1 = L1  # 첫 번째 링크 길이
        self.L2 = L2  # 두 번째 링크 길이
        
        # 그래프 설정 - 더 큰 창과 두 개의 서브플롯
        self.fig_robot, (self.ax_robot, self.ax_trajectory) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 로봇 팔 시각화 (왼쪽)
        self.ax_robot.set_xlim(-2.5, 2.5)
        self.ax_robot.set_ylim(-2.5, 2.5)
        self.ax_robot.set_aspect('equal')
        self.ax_robot.grid(True)
        self.ax_robot.set_title('2-DOF Robot Arm Motion')
        
        # 로봇 팔 구성 요소
        self.link1, = self.ax_robot.plot([], [], 'b-', linewidth=8, label='Link 1')
        self.link2, = self.ax_robot.plot([], [], 'r-', linewidth=6, label='Link 2')
        self.joint1 = Circle((0, 0), 0.1, color='black')
        self.joint2 = Circle((0, 0), 0.08, color='black')
        self.end_effector = Circle((0, 0), 0.06, color='green')
        
        self.ax_robot.add_patch(self.joint1)
        self.ax_robot.add_patch(self.joint2)
        self.ax_robot.add_patch(self.end_effector)
        
        # 궤적 그리기 (로봇 팔 창)
        self.trajectory, = self.ax_robot.plot([], [], 'g--', alpha=0.7, linewidth=2, label='Actual Trajectory')
        self.ref_trajectory, = self.ax_robot.plot([], [], 'k:', alpha=0.7, linewidth=2, label='Reference Trajectory')
        
        self.ax_robot.legend()
        
        # 확대된 궤적 시각화 (오른쪽)
        self.ax_trajectory.set_aspect('equal')
        self.ax_trajectory.grid(True)
        self.ax_trajectory.set_title('End-Effector Trajectory (Zoomed)')
        self.ax_trajectory.set_xlabel('X Position (m)')
        self.ax_trajectory.set_ylabel('Y Position (m)')
        
        # 확대된 궤적 라인
        self.trajectory_zoom, = self.ax_trajectory.plot([], [], 'g-', linewidth=3, label='Actual Trajectory')
        self.ref_trajectory_zoom, = self.ax_trajectory.plot([], [], 'r--', linewidth=3, label='Reference Trajectory')
        self.current_pos = self.ax_trajectory.plot([], [], 'bo', markersize=8, label='Current Position')[0]
        self.ref_pos = self.ax_trajectory.plot([], [], 'rs', markersize=8, label='Reference Position')[0]
        
        self.ax_trajectory.legend()
        
        # 궤적 데이터 저장 - 크기 제한
        self.max_points = 10000  # 궤적 점 개수 제한
        self.actual_traj_x = []
        self.actual_traj_y = []
        self.ref_traj_x = []
        self.ref_traj_y = []
        
        # 업데이트 카운터 (성능 최적화)
        self.update_counter = 0
        self.update_interval = 3  # 3번에 1번만 업데이트
    
    def update_robot_pose(self, q, q_ref):
        """로봇 자세 업데이트 - 성능 최적화"""
        self.update_counter += 1
        
        q1, q2 = q[0], q[1]
        q1_ref, q2_ref = q_ref[0], q_ref[1]
        
        # 실제 위치 계산
        x1 = self.L1 * np.cos(q1)
        y1 = self.L1 * np.sin(q1)
        x2 = x1 + self.L2 * np.cos(q1 + q2)
        y2 = y1 + self.L2 * np.sin(q1 + q2)
        
        # 참조 위치 계산
        x1_ref = self.L1 * np.cos(q1_ref)
        y1_ref = self.L1 * np.sin(q1_ref)
        x2_ref = x1_ref + self.L2 * np.cos(q1_ref + q2_ref)
        y2_ref = y1_ref + self.L2 * np.sin(q1_ref + q2_ref)
        
        # 링크 업데이트 (매번)
        self.link1.set_data([0, x1], [0, y1])
        self.link2.set_data([x1, x2], [y1, y2])
        
        # 조인트 위치 업데이트 (매번)
        self.joint1.center = (0, 0)
        self.joint2.center = (x1, y1)
        self.end_effector.center = (x2, y2)
        
        # 궤적 데이터 추가 (매번)
        self.actual_traj_x.append(x2)
        self.actual_traj_y.append(y2)
        self.ref_traj_x.append(x2_ref)
        self.ref_traj_y.append(y2_ref)
        
        # 데이터 크기 제한 (성능 최적화)
        if len(self.actual_traj_x) > self.max_points:
            self.actual_traj_x = self.actual_traj_x[-self.max_points:]
            self.actual_traj_y = self.actual_traj_y[-self.max_points:]
            self.ref_traj_x = self.ref_traj_x[-self.max_points:]
            self.ref_traj_y = self.ref_traj_y[-self.max_points:]
        
        # 궤적 업데이트 (간격을 두고)
        if self.update_counter % self.update_interval == 0:
            # 궤적 라인 업데이트
            self.trajectory.set_data(self.actual_traj_x, self.actual_traj_y)
            self.ref_trajectory.set_data(self.ref_traj_x, self.ref_traj_y)
            
            # 확대된 궤적 창 업데이트
            self.trajectory_zoom.set_data(self.actual_traj_x, self.actual_traj_y)
            self.ref_trajectory_zoom.set_data(self.ref_traj_x, self.ref_traj_y)
            self.current_pos.set_data([x2], [y2])
            self.ref_pos.set_data([x2_ref], [y2_ref])
            
            # 확대된 궤적 창의 축 범위 고정 (로봇 팔 작업 공간 기준)
            margin = 0.1
            self.ax_trajectory.set_xlim(-2.0 - margin, 2.0 + margin)  # 로봇 팔 최대 도달 범위 기준
            self.ax_trajectory.set_ylim(-2.0 - margin, 2.0 + margin)
            
            # 화면 업데이트
            self.fig_robot.canvas.draw_idle()  # draw() 대신 draw_idle() 사용
            self.fig_robot.canvas.flush_events()

def show_trajectory_detail(states, references, t):
    """궤적 세부 사항을 별도 창에 표시"""
    L1, L2 = 1.0, 0.8
    
    # 끝점 위치 계산
    actual_x = []
    actual_y = []
    ref_x = []
    ref_y = []
    
    for i in range(len(states)):
        q1, q2 = states[i, 0], states[i, 1]
        q1_ref, q2_ref = references[i, 0], references[i, 1]
        
        # 실제 끝점 위치
        x1 = L1 * np.cos(q1)
        y1 = L1 * np.sin(q1)
        x2 = x1 + L2 * np.cos(q1 + q2)
        y2 = y1 + L2 * np.sin(q1 + q2)
        
        # 참조 끝점 위치
        x1_ref = L1 * np.cos(q1_ref)
        y1_ref = L1 * np.sin(q1_ref)
        x2_ref = x1_ref + L2 * np.cos(q1_ref + q2_ref)
        y2_ref = y1_ref + L2 * np.sin(q1_ref + q2_ref)
        
        actual_x.append(x2)
        actual_y.append(y2)
        ref_x.append(x2_ref)
        ref_y.append(y2_ref)
    
    # 궤적 세부 시각화
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 전체 궤적
    ax1.plot(actual_x, actual_y, 'b-', linewidth=2, label='Actual Trajectory')
    ax1.plot(ref_x, ref_y, 'r--', linewidth=2, label='Reference Trajectory')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Complete End-Effector Trajectory')
    ax1.grid(True)
    ax1.axis('equal')
    
    # X 위치 vs 시간
    ax2.plot(t, actual_x, 'b-', linewidth=2, label='Actual X')
    ax2.plot(t, ref_x, 'r--', linewidth=2, label='Reference X')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('X Position (m)')
    ax2.set_title('X Position vs Time')
    ax2.grid(True)
    ax2.legend()
    
    # Y 위치 vs 시간
    ax3.plot(t, actual_y, 'b-', linewidth=2, label='Actual Y')
    ax3.plot(t, ref_y, 'r--', linewidth=2, label='Reference Y')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Y Position (m)')
    ax3.set_title('Y Position vs Time')
    ax3.grid(True)
    ax3.legend()
    
    # 추적 오차 (위치)
    pos_error = np.sqrt((np.array(actual_x) - np.array(ref_x))**2 + 
                       (np.array(actual_y) - np.array(ref_y))**2)
    ax4.plot(t, pos_error, 'g-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position Error (m)')
    ax4.set_title('End-Effector Position Error')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 창이 닫히지 않도록 대기
    input("Press Enter to continue...")
    
    print(f"Maximum position error: {np.max(pos_error):.6f} m")
    print(f"Average position error: {np.mean(pos_error):.6f} m")
    print(f"Final position error: {pos_error[-1]:.6f} m")

def plot_results(t, states, controls, tracking_errors, references, f_est, d_est, controller):
    """기본 추적 성능 결과 플롯"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Tracking Performance Analysis', fontsize=16)
    
    # 추적 성능
    axes[0, 0].plot(t, states[:, 0], 'b-', label='q1 (actual)', linewidth=2)
    axes[0, 0].plot(t, references[:, 0], 'r--', label='q1 (reference)', linewidth=2)
    axes[0, 0].set_title('Joint 1 Tracking Performance')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angle (rad)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[1, 0].plot(t, states[:, 1], 'b-', label='q2 (actual)', linewidth=2)
    axes[1, 0].plot(t, references[:, 1], 'r--', label='q2 (reference)', linewidth=2)
    axes[1, 0].set_title('Joint 2 Tracking Performance')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Angle (rad)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 추적 오차
    axes[0, 1].plot(t, tracking_errors[:, 0], 'g-', linewidth=2)
    axes[0, 1].set_title('Joint 1 Tracking Error')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Error (rad)')
    axes[0, 1].grid(True)
    
    axes[1, 1].plot(t, tracking_errors[:, 1], 'g-', linewidth=2)
    axes[1, 1].set_title('Joint 2 Tracking Error')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Error (rad)')
    axes[1, 1].grid(True)
    
    # 제어 입력
    axes[0, 2].plot(t, controls[:, 0], 'm-', linewidth=2)
    axes[0, 2].set_title('Joint 1 Control Input')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Torque (N⋅m)')
    axes[0, 2].grid(True)
    
    axes[1, 2].plot(t, controls[:, 1], 'm-', linewidth=2)
    axes[1, 2].set_title('Joint 2 Control Input')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Torque (N⋅m)')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 창이 닫히지 않도록 대기
    input("Press Enter to continue...")
    
    # 성능 지표 출력
    print("\n=== Tracking Performance Metrics ===")
    if len(tracking_errors) > 1000:
        max_error_1 = np.max(np.abs(tracking_errors[-1000:, 0]))  # 마지막 10초
        max_error_2 = np.max(np.abs(tracking_errors[-1000:, 1]))
        print(f"Joint 1 Max Steady-State Error: {max_error_1:.6f} rad")
        print(f"Joint 2 Max Steady-State Error: {max_error_2:.6f} rad")
    
    rms_error_1 = np.sqrt(np.mean(tracking_errors[:, 0]**2))
    rms_error_2 = np.sqrt(np.mean(tracking_errors[:, 1]**2))
    print(f"Joint 1 RMS Tracking Error: {rms_error_1:.6f} rad")
    print(f"Joint 2 RMS Tracking Error: {rms_error_2:.6f} rad")
    print(f"Average Control Input Magnitude: {np.mean(np.abs(controls)):.4f} N⋅m")

def show_disturbance_analysis(t, disturbances, d_hats, controller):
    """외란 분석 결과 표시"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Disturbance Observer Analysis', fontsize=16)
    
    # Joint 1 외란
    axes[0, 0].plot(t, disturbances[:, 0], 'r-', linewidth=2, label='Actual Disturbance')
    axes[0, 0].plot(t, d_hats[:, 0], 'b--', linewidth=2, label='Estimated Disturbance')
    axes[0, 0].set_title('Joint 1 Disturbance')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Disturbance (N⋅m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Joint 2 외란
    axes[1, 0].plot(t, disturbances[:, 1], 'r-', linewidth=2, label='Actual Disturbance')
    axes[1, 0].plot(t, d_hats[:, 1], 'b--', linewidth=2, label='Estimated Disturbance')
    axes[1, 0].set_title('Joint 2 Disturbance')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Disturbance (N⋅m)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 외란 추정 오차
    d_error1 = disturbances[:, 0] - d_hats[:, 0]
    d_error2 = disturbances[:, 1] - d_hats[:, 1]
    
    axes[0, 1].plot(t, d_error1, 'g-', linewidth=2)
    axes[0, 1].set_title('Joint 1 Disturbance Estimation Error')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Error (N⋅m)')
    axes[0, 1].grid(True)
    
    axes[1, 1].plot(t, d_error2, 'g-', linewidth=2)
    axes[1, 1].set_title('Joint 2 Disturbance Estimation Error')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Error (N⋅m)')
    axes[1, 1].grid(True)
    
    # 학습 곡선
    axes[0, 2].plot(controller.actor_losses, 'b-', linewidth=2)
    axes[0, 2].set_title('Actor Network Loss')
    axes[0, 2].set_xlabel('Training Steps')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True)
    axes[0, 2].set_yscale('symlog')
    
    axes[1, 2].plot(controller.critic_losses, 'r-', linewidth=2)
    axes[1, 2].set_title('Critic Network Loss')
    axes[1, 2].set_xlabel('Training Steps')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].grid(True)
    axes[1, 2].set_yscale('symlog')
    
    plt.tight_layout()
    plt.show()
    
    # 창이 닫히지 않도록 대기
    input("Press Enter to continue...")
    
    # 성능 지표 출력
    print("\n=== Disturbance Observer Performance ===")
    rms_error1 = np.sqrt(np.mean(d_error1**2))
    rms_error2 = np.sqrt(np.mean(d_error2**2))
    max_error1 = np.max(np.abs(d_error1))
    max_error2 = np.max(np.abs(d_error2))
    
    print(f"Joint 1 - RMS estimation error: {rms_error1:.4f} N⋅m")
    print(f"Joint 1 - Max estimation error: {max_error1:.4f} N⋅m")
    print(f"Joint 2 - RMS estimation error: {rms_error2:.4f} N⋅m")
    print(f"Joint 2 - Max estimation error: {max_error2:.4f} N⋅m")

def simulate_system_with_visualization():
    """시각화와 함께 시스템 시뮬레이션 - 성능 최적화"""
    controller = DisturbanceObserverActorCriticController()
    
    # 시뮬레이션 파라미터
    t_span = (0, 15)  # 시각화를 위해 시간 단축
    dt = 0.02  # 시각화를 위해 시간 간격 증가
    
    # 로봇 팔 시각화만 (실시간)
    robot_viz = RobotArm2DVisualizer()
    
    # 초기 조건
    initial_state = np.array([0.1, 0.1, 0.0, 0.0])
    state = initial_state.copy()
    
    # 결과 저장용 (plot_results를 위해)
    states = []
    controls = []
    tracking_errors = []
    references = []
    f_estimations = []
    d_estimations = []
    disturbances = []
    d_hats = []
    
    # 시뮬레이션 실행
    print("Starting real-time robot arm visualization...")
    print("Detailed analysis will be shown after simulation completes.")
    print("Close the robot arm window to stop simulation.")
    
    t_eval = np.arange(0, t_span[1], dt)
    
    # 진행 상황 표시
    progress_bar = tqdm(total=len(t_eval), desc="Simulation Progress")
    
    for i, t in enumerate(t_eval):
        try:
            # 제어 입력 계산
            u, z1, z2, f_hat, d_hat = controller.control_law(t, state, dt)
            
            # 시스템 동역학 적분
            def dynamics_wrapper(t, x):
                return controller.robot.dynamics(t, x, u)
            
            sol = solve_ivp(dynamics_wrapper, [t, t + dt], state, dense_output=True)
            state = sol.y[:, -1]
            
            # 참조 궤적
            yr, _, _ = controller.reference_trajectory(t)
            
            # 실제 외란 값 계산
            actual_disturbance = controller.robot.external_disturbance(t)
            
            # 손실 값 가져오기
            actor_loss = controller.actor_losses[-1] if controller.actor_losses else 0
            critic_loss = controller.critic_losses[-1] if controller.critic_losses else 0
            
            # 로봇 팔 시각화만 업데이트
            robot_viz.update_robot_pose(state[:2], yr)
            
            # 진행 상황 업데이트 (10번에 1번만)
            if i % 10 == 0:
                progress_bar.update(10)
                progress_bar.set_postfix({
                    'E1': f'{z1[0]:.4f}',
                    'E2': f'{z1[1]:.4f}',
                    'D1': f'{actual_disturbance[0]:.2f}',
                    'D2': f'{actual_disturbance[1]:.2f}',
                    'A_Loss': f'{actor_loss:.3f}',
                    'C_Loss': f'{critic_loss:.3f}'
                })
            
            # 데이터 저장 (plot_results를 위해)
            states.append(state.copy())
            controls.append(u.copy())
            tracking_errors.append(z1.copy())
            references.append(yr.copy())
            f_estimations.append(f_hat.copy())
            d_estimations.append(d_hat.copy())
            disturbances.append(actual_disturbance.copy())
            d_hats.append(d_hat.copy())
            
            # 신경망 데이터 크기 제한 (메모리 절약)
            max_loss_history = 2000
            if len(controller.actor_losses) > max_loss_history:
                controller.actor_losses = controller.actor_losses[-max_loss_history:]
            if len(controller.critic_losses) > max_loss_history:
                controller.critic_losses = controller.critic_losses[-max_loss_history:]
            
            # 사용자 인터럽트 체크
            if not plt.get_fignums():
                print("User closed the robot arm window. Stopping simulation.")
                break
                
        except KeyboardInterrupt:
            print("\nUser interrupted the simulation.")
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            break
    
    progress_bar.close()
    
    # interactive mode 비활성화
    plt.ioff()
    
    print("\nSimulation completed! Generating detailed analysis...")
    
    # 데이터 배열 변환
    t_array = np.array(t_eval[:len(states)])
    states_array = np.array(states)
    controls_array = np.array(controls)
    errors_array = np.array(tracking_errors)
    refs_array = np.array(references)
    f_est_array = np.array(f_estimations)
    d_est_array = np.array(d_estimations)
    disturbances_array = np.array(disturbances)
    d_hats_array = np.array(d_hats)
    
    # 1. 기본 추적 성능 결과
    plot_results(t_array, states_array, controls_array, errors_array, refs_array, f_est_array, d_est_array, controller)
    
    # 2. 확대된 경로 시각화
    show_trajectory_detail(states_array, refs_array, t_array)
    
    # 3. 외란 분석 (새로 추가)
    show_disturbance_analysis(t_array, disturbances_array, d_hats_array, controller)
    
    print("All analysis completed!")
    return controller

def simulate_system():
    """시스템 시뮬레이션"""
    controller = DisturbanceObserverActorCriticController()
    
    # 시뮬레이션 파라미터
    t_span = (0, 30)
    dt = 0.01
    t_eval = np.arange(0, 30, dt)
    
    # 초기 조건
    initial_state = np.array([0.1, 0.1, 0.0, 0.0])  # [q1, q2, q1_dot, q2_dot]
    
    # 결과 저장용
    states = []
    controls = []
    tracking_errors = []
    references = []
    f_estimations = []
    d_estimations = []
    
    state = initial_state.copy()
    
    for i, t in enumerate(t_eval):
        # 제어 입력 계산
        u, z1, z2, f_hat, d_hat = controller.control_law(t, state, dt)
        
        # 시스템 동역학 적분
        def dynamics_wrapper(t, x):
            return controller.robot.dynamics(t, x, u)
        
        sol = solve_ivp(dynamics_wrapper, [t, t + dt], state, dense_output=True)
        state = sol.y[:, -1]
        
        # 참조 궤적
        yr, _, _ = controller.reference_trajectory(t)
        
        # 결과 저장
        states.append(state.copy())
        controls.append(u.copy())
        tracking_errors.append(z1.copy())
        references.append(yr.copy())
        f_estimations.append(f_hat.copy())
        d_estimations.append(d_hat.copy())
    
    return (np.array(t_eval), np.array(states), np.array(controls), 
            np.array(tracking_errors), np.array(references),
            np.array(f_estimations), np.array(d_estimations), controller)

class ActorCriticController:
    """액터-크리틱 제어기 (외란 관측기 없음)"""
    def __init__(self):
        self.robot = RobotManipulator()
        
        # 신경망 초기화
        self.actor = ActorNetwork()
        self.critic = CriticNetwork()
        
        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.01)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.02)
        
        # 제어 파라미터 (논문 값)
        self.k1 = 30
        self.k2 = 10
        self.kn = 10
        self.beta = 100
        
        # 학습 파라미터
        self.ka1 = 20
        self.ka2 = 1
        self.ka3 = 5
        self.kc1 = 2
        self.kc2 = 0.1
        self.gamma = 0.1  # 할인 인자
        
        # 성능 지수 가중치
        self.Q = np.diag([50, 200])
        self.R = np.diag([0.1, 0.1])
        
        # 저장용 변수
        self.actor_losses = []
        self.critic_losses = []
        
    def reference_trajectory(self, t):
        """참조 궤적"""
        yr1 = 0.6 * np.sin(3.14 * t) * (1 - np.exp(-t))
        yr2 = 0.8 * np.sin(3.14 * t) * (1 - np.exp(-t))
        
        # 1차, 2차 미분
        yr1_dot = 0.6 * (3.14 * np.cos(3.14 * t) * (1 - np.exp(-t)) + np.sin(3.14 * t) * np.exp(-t))
        yr2_dot = 0.8 * (3.14 * np.cos(3.14 * t) * (1 - np.exp(-t)) + np.sin(3.14 * t) * np.exp(-t))
        
        yr1_ddot = 0.6 * (-3.14**2 * np.sin(3.14 * t) * (1 - np.exp(-t)) + 
                          2 * 3.14 * np.cos(3.14 * t) * np.exp(-t) - np.sin(3.14 * t) * np.exp(-t))
        yr2_ddot = 0.8 * (-3.14**2 * np.sin(3.14 * t) * (1 - np.exp(-t)) + 
                          2 * 3.14 * np.cos(3.14 * t) * np.exp(-t) - np.sin(3.14 * t) * np.exp(-t))
        
        return np.array([yr1, yr2]), np.array([yr1_dot, yr2_dot]), np.array([yr1_ddot, yr2_ddot])
    
    def compute_filtered_errors(self, state, yr, yr_dot, yr_ddot):
        """필터링된 추적 오차 계산"""
        q = state[:2]
        q_dot = state[2:]
        
        z1 = q - yr
        z2 = q_dot - yr_dot + self.k1 * z1
        
        return z1, z2
    
    def compute_cost(self, z1, u):
        """즉시 비용 함수"""
        return z1.T @ self.Q @ z1 + u.T @ self.R @ u
    
    def update_networks(self, state, z1, z2, u, reward, next_state, dt):
        """액터-크리틱 네트워크 업데이트"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        z1_tensor = torch.FloatTensor(z1).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        next_z1_tensor = torch.FloatTensor(next_state[:2] - self.reference_trajectory(0)[0]).unsqueeze(0)
        
        # 크리틱 네트워크 업데이트
        current_value = self.critic(z1_tensor)
        next_value = self.critic(next_z1_tensor)
        target_value = reward + self.gamma * next_value
        
        critic_loss = nn.MSELoss()(current_value, target_value.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 액터 네트워크 업데이트
        f_hat = self.actor(state_tensor)
        
        # 복합 적응 법칙 (예측 오차 + 모델링 오차)
        prediction_error = current_value.item()
        modeling_error = np.linalg.norm(z2)  # 모델링 오차 근사
        
        actor_loss = prediction_error + 0.1 * modeling_error**2
        actor_loss_tensor = torch.tensor(actor_loss, requires_grad=True)
        
        self.actor_optimizer.zero_grad()
        actor_loss_tensor.backward()
        self.actor_optimizer.step()
        
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss.item())
        
        return f_hat.detach().numpy().flatten()
    
    def control_law(self, t, state, dt):
        """제어 법칙 (외란 관측기 없음)"""
        q = state[:2]
        q_dot = state[2:]
        
        # 참조 궤적
        yr, yr_dot, yr_ddot = self.reference_trajectory(t)
        
        # 필터링된 오차
        z1, z2 = self.compute_filtered_errors(state, yr, yr_dot, yr_ddot)
        
        # 액터 네트워크로 미지 동역학 추정
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            f_hat = self.actor(state_tensor).numpy().flatten()
        
        # 제어 입력 계산 (외란 관측기 없이)
        # u = M(q)[yr_ddot - k*z - kn*zn - f_hat]
        k_z = self.k1 * z1 + self.k2 * z2
        u = self.robot.mass_matrix(q) @ (yr_ddot - k_z - self.kn * z2 - f_hat)
        
        # 제어 입력 제한
        u = np.clip(u, -50, 50)
        
        # 네트워크 업데이트 (강화학습)
        reward = -self.compute_cost(z1, u)
        next_state = state.copy()  # 간단한 근사
        f_hat_updated = self.update_networks(state, z1, z2, u, reward, next_state, dt)
        
        return u, z1, z2, f_hat_updated, np.zeros(2)  # d_hat은 0으로 반환

def compare_controllers():
    """두 제어기 비교 시뮬레이션"""
    print("Comparing Actor-Critic vs DOB+Actor-Critic controllers...")
    
    # 시뮬레이션 파라미터
    t_span = (0, 15)
    dt = 0.02
    t_eval = np.arange(0, t_span[1], dt)
    initial_state = np.array([0.1, 0.1, 0.0, 0.0])
    
    # 두 제어기 초기화
    ac_controller = ActorCriticController()
    dob_ac_controller = DisturbanceObserverActorCriticController()
    
    # 결과 저장용
    results = {
        'ac': {'states': [], 'controls': [], 'errors': [], 'disturbances': []},
        'dob_ac': {'states': [], 'controls': [], 'errors': [], 'disturbances': []}
    }
    
    print("Running Actor-Critic only simulation...")
    # 액터-크리틱만 시뮬레이션
    state_ac = initial_state.copy()
    for i, t in enumerate(tqdm(t_eval, desc="AC Only")):
        u, z1, z2, f_hat, _ = ac_controller.control_law(t, state_ac, dt)
        
        def dynamics_wrapper(t, x):
            return ac_controller.robot.dynamics(t, x, u)
        
        sol = solve_ivp(dynamics_wrapper, [t, t + dt], state_ac, dense_output=True)
        state_ac = sol.y[:, -1]
        
        yr, _, _ = ac_controller.reference_trajectory(t)
        actual_disturbance = ac_controller.robot.external_disturbance(t)
        
        results['ac']['states'].append(state_ac.copy())
        results['ac']['controls'].append(u.copy())
        results['ac']['errors'].append(z1.copy())
        results['ac']['disturbances'].append(actual_disturbance.copy())
    
    print("Running DOB+Actor-Critic simulation...")
    # DOB + 액터-크리틱 시뮬레이션
    state_dob = initial_state.copy()
    for i, t in enumerate(tqdm(t_eval, desc="DOB+AC")):
        u, z1, z2, f_hat, d_hat = dob_ac_controller.control_law(t, state_dob, dt)
        
        def dynamics_wrapper(t, x):
            return dob_ac_controller.robot.dynamics(t, x, u)
        
        sol = solve_ivp(dynamics_wrapper, [t, t + dt], state_dob, dense_output=True)
        state_dob = sol.y[:, -1]
        
        yr, _, _ = dob_ac_controller.reference_trajectory(t)
        actual_disturbance = dob_ac_controller.robot.external_disturbance(t)
        
        results['dob_ac']['states'].append(state_dob.copy())
        results['dob_ac']['controls'].append(u.copy())
        results['dob_ac']['errors'].append(z1.copy())
        results['dob_ac']['disturbances'].append(actual_disturbance.copy())
    
    # 결과 변환
    for key in results:
        for subkey in results[key]:
            results[key][subkey] = np.array(results[key][subkey])
    
    # 비교 결과 플롯
    plot_comparison_results(t_eval, results, ac_controller, dob_ac_controller)
    
    return results, ac_controller, dob_ac_controller

def plot_comparison_results(t, results, ac_controller, dob_ac_controller):
    """비교 결과 플롯"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Actor-Critic vs DOB+Actor-Critic Comparison', fontsize=16)
    
    # Joint 1 추적 성능
    axes[0, 0].plot(t, results['ac']['states'][:, 0], 'b-', label='AC Only', linewidth=2)
    axes[0, 0].plot(t, results['dob_ac']['states'][:, 0], 'r-', label='DOB+AC', linewidth=2)
    yr = [ac_controller.reference_trajectory(time)[0][0] for time in t]
    axes[0, 0].plot(t, yr, 'k--', label='Reference', linewidth=1, alpha=0.7)
    axes[0, 0].set_title('Joint 1 Tracking Performance')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angle (rad)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Joint 2 추적 성능
    axes[0, 1].plot(t, results['ac']['states'][:, 1], 'b-', label='AC Only', linewidth=2)
    axes[0, 1].plot(t, results['dob_ac']['states'][:, 1], 'r-', label='DOB+AC', linewidth=2)
    yr = [ac_controller.reference_trajectory(time)[0][1] for time in t]
    axes[0, 1].plot(t, yr, 'k--', label='Reference', linewidth=1, alpha=0.7)
    axes[0, 1].set_title('Joint 2 Tracking Performance')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angle (rad)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 추적 오차 비교
    axes[1, 0].plot(t, results['ac']['errors'][:, 0], 'b-', label='AC Only', linewidth=2)
    axes[1, 0].plot(t, results['dob_ac']['errors'][:, 0], 'r-', label='DOB+AC', linewidth=2)
    axes[1, 0].set_title('Joint 1 Tracking Error Comparison')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Error (rad)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(t, results['ac']['errors'][:, 1], 'b-', label='AC Only', linewidth=2)
    axes[1, 1].plot(t, results['dob_ac']['errors'][:, 1], 'r-', label='DOB+AC', linewidth=2)
    axes[1, 1].set_title('Joint 2 Tracking Error Comparison')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Error (rad)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 제어 입력 비교
    axes[2, 0].plot(t, results['ac']['controls'][:, 0], 'b-', label='AC Only', linewidth=2)
    axes[2, 0].plot(t, results['dob_ac']['controls'][:, 0], 'r-', label='DOB+AC', linewidth=2)
    axes[2, 0].set_title('Joint 1 Control Input Comparison')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Torque (N⋅m)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(t, results['ac']['controls'][:, 1], 'b-', label='AC Only', linewidth=2)
    axes[2, 1].plot(t, results['dob_ac']['controls'][:, 1], 'r-', label='DOB+AC', linewidth=2)
    axes[2, 1].set_title('Joint 2 Control Input Comparison')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Torque (N⋅m)')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 창이 닫히지 않도록 대기
    input("Press Enter to continue...")
    
    # 성능 지표 비교
    print("\n=== Performance Comparison ===")
    
    # RMS 오차 계산
    ac_rms_1 = np.sqrt(np.mean(results['ac']['errors'][:, 0]**2))
    ac_rms_2 = np.sqrt(np.mean(results['ac']['errors'][:, 1]**2))
    dob_ac_rms_1 = np.sqrt(np.mean(results['dob_ac']['errors'][:, 0]**2))
    dob_ac_rms_2 = np.sqrt(np.mean(results['dob_ac']['errors'][:, 1]**2))
    
    print(f"Actor-Critic Only:")
    print(f"  Joint 1 RMS Error: {ac_rms_1:.6f} rad")
    print(f"  Joint 2 RMS Error: {ac_rms_2:.6f} rad")
    print(f"  Average Control Effort: {np.mean(np.abs(results['ac']['controls'])):.4f} N⋅m")
    
    print(f"\nDOB + Actor-Critic:")
    print(f"  Joint 1 RMS Error: {dob_ac_rms_1:.6f} rad")
    print(f"  Joint 2 RMS Error: {dob_ac_rms_2:.6f} rad")
    print(f"  Average Control Effort: {np.mean(np.abs(results['dob_ac']['controls'])):.4f} N⋅m")
    
    print(f"\nImprovement with DOB:")
    print(f"  Joint 1 Error Reduction: {(1 - dob_ac_rms_1/ac_rms_1)*100:.2f}%")
    print(f"  Joint 2 Error Reduction: {(1 - dob_ac_rms_2/ac_rms_2)*100:.2f}%")

if __name__ == "__main__":
    print("Disturbance Observer-based Actor-Critic Control Simulation")
    print("1: Real-time Visualization (DOB+AC)")
    print("2: Basic Simulation (DOB+AC)")
    print("3: Compare AC vs DOB+AC")
    
    choice = input("Select option (1, 2, or 3): ").strip()
    
    # interactive mode 설정
    if choice == "1":
        plt.ion()  # 실시간 시각화용
        controller = simulate_system_with_visualization()
    elif choice == "2":
        plt.ioff()  # 정적 그래프용
        print("Starting basic simulation...")
        t, states, controls, errors, refs, f_est, d_est, controller = simulate_system()
        plot_results(t, states, controls, errors, refs, f_est, d_est, controller)
    elif choice == "3":
        plt.ioff()  # 정적 그래프용
        results, ac_controller, dob_ac_controller = compare_controllers()
    else:
        plt.ioff()  # 정적 그래프용
        print("Invalid choice, running comparison by default...")
        results, ac_controller, dob_ac_controller = compare_controllers()
    
    print("All simulations completed!")