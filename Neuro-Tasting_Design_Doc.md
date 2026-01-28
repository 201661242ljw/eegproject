# **Neuro-Tasting 系统交互升级设计文档 (v1.0)**

**日期**: 2026-01-25

**状态**: 规划中

**目标**: 将工程原型转化为具备完整交互流程的桌面应用程序。

## **1\. 背景与目标 (Background & Goals)**

目前系统为“自动启动、自动连接、无交互”的纯展示程序。为了提升可用性和产品化程度，需要引入用户交互层。

**核心目标**：

1. **增加用户控制**：允许用户控制连接、开始、结束。  
2. **数据归档**：通过用户标记 (User ID) 实现数据的可追溯性。  
3. **流程完整性**：构建“准备 \-\> 采集 \-\> 结算 \-\> 报告”的完整闭环。  
4. **视觉层级化**：利用模块化设计解决视觉杂乱问题。

## **2\. 用户流程 (User Flow)**

### **2.1 阶段一：初始/仪表盘界面 (Dashboard)**

程序启动后的默认界面。用户在此处完成所有准备工作。

* **输入信息**：  
  * 输入框：User ID / Name (必填/选填，默认为 "Guest")。  
  * 用于生成文件名：YYYY\_MM\_DD\_HHMM\_\[UserID\].csv。  
* **硬件自检**：  
  * 后台自动扫描蓝牙设备。  
  * **状态指示灯**：  
    * 🔴 红色：未找到设备 / 蓝牙关闭。  
    * 🟡 黄色：扫描中...  
    * 🟢 绿色：设备就绪 (检测到 "RFstar\_7FEA")。  
* **历史记录 (Sidebar)**：  
  * 左侧列表显示最近 5 次采集记录 (CSV)。  
  * 点击可进入“回放模式”。  
* **操作**：  
  * \[开始采集\] 按钮：仅当蓝牙状态为 🟢 时可用。

### **2.2 阶段二：实时采集界面 (Session Live)**

点击开始后进入的沉浸式界面。

* **视觉布局 (UI Layout)**：采用“HUD 面板”风格。  
  * **底层**：保留全屏动态背景、螺旋柱状图、3D 大脑模型（无边框，沉浸感）。  
  * **左上悬浮窗**：原始脑电波形 (Raw EEG)。  
  * **右上悬浮窗**：频谱分析 (PSD) & 频段能量。  
  * **左下悬浮窗**：感官指标 (Metrics Dashboard)。  
  * **右下悬浮窗**：2D 情感状态空间 (Valence-Arousal) & 实时趋势。  
* **交互控件**：  
  * \[停止/结算\] 按钮：位于角落，点击后结束采集并保存。  
  * \[事件标记\] (Mark)：(可选) 记录当前时间点的特殊事件（如“品尝”）。  
  * 键盘快捷键：L 切换频谱对数坐标，ESC 停止。

### **2.3 阶段三：结算与报告界面 (Report) \- *后续规划***

采集结束后的总结页面。

* **内容**：  
  * 本次采集时长、数据质量评分。  
  * 情绪状态分布饼图 (Flow/Relax/Anxiety/Boredom)。  
  * 关键指标平均值。  
* **操作**：  
  * \[打开 CSV 文件夹\]  
  * \[生成 PDF 报告\]  
  * \[返回主页\]

## **3\. UI/UX 详细设计 (UI Specifications)**

**风格定义**：Cyberpunk / Sci-Fi Interface

**配色方案**：

* 背景：深蓝黑 (\#0A0A0F)  
* 面板背景：半透明黑玻璃 (\#101520, Alpha=200)  
* 边框：暗青色 (\#28323C)  
* 高亮/交互：霓虹金 (\#FFD700)、赛博青 (\#00FFFF)

**模块化设计 (Modular Layout)**：

不要将文字直接写在背景上。所有数据展示必须包含在统一风格的 Panel 类容器中：

* **Header**：小字体标题，带装饰性线条。  
* **Body**：内容区域。  
* **Border**：细线边框，四角可能有装饰性加粗 (Corner Brackets)。

## **4\. 技术架构 (Technical Architecture)**

为了实现界面切换，必须从单一循环重构为**状态机模式 (State Machine)**。

### **4.1 核心类结构**

* GameApp: 主程序入口，管理 Pygame 窗口和主循环。  
  * state: 当前状态 (MENU, RUNNING, REPORT)。  
* Scene: 抽象基类，所有界面继承此类。  
  * handle\_input(events)  
  * update()  
  * draw(screen)  
* SceneMenu (Scene): 实现输入框、蓝牙扫描线程、按钮逻辑。  
* SceneSession (Scene): 封装原 main\_v16.py 的可视化逻辑，负责数据接收与绘制。  
* UIManager: 简单的 UI 库，包含 Button, TextInput, Panel。

### **4.2 数据流 (Data Flow)**

1. **Menu**:  
   * BluetoothScanner 线程运行 \-\> 更新 global\_bt\_status。  
   * 用户输入 \-\> 存储在 session\_config 字典。  
2. **Transition**:  
   * 点击 Start \-\> 实例化 SceneSession，将 session\_config 传入。  
   * SceneSession 启动 EEGBluetoothReceiver 线程。  
3. **Running**:  
   * AsyncProcessor 处理数据 \-\> 放入 UI\_Queue。  
   * UI 从 Queue 取数据渲染。  
   * 同时写入 CSV 文件。  
4. **Stop**:  
   * 关闭 CSV，停止蓝牙线程。  
   * 切换回 SceneMenu (或 SceneReport)。

## **5\. 待办事项 (Implementation Roadmap)**

* \[x\] **Step 1**: 编写 main\_ui.py，建立状态机框架，实现 UI 控件库（按钮、输入框）。  
* \[ \] **Step 2**: 在 Menu 中集成蓝牙后台扫描逻辑。  
* \[ \] **Step 3**: 将 main\_v16.py 的绘图逻辑移植到 SceneSession 中。  
* \[ \] **Step 4**: 实现 PDF 报告生成功能。