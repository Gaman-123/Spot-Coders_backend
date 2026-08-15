<div align="center">

<br/>

```
╔╦╗╔═╗╔╦╗╔═╗╔═╗╔╗╔╔╗╔╔═╗╔═╗╔╦╗
║║║║╣  ║║║  ║ ║║║║║║║║╣ ║   ║
╩ ╩╚═╝═╩╝╚═╝╚═╝╝╚╝╝╚╝╚═╝╚═╝ ╩
```

# MedConnect — Doctor Clinical Interface

**A professional, open-source clinical suite for modern healthcare providers.**

[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=flat-square&logo=vite)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](CONTRIBUTING.md)

</div>

---

## 📋 Overview

**MedConnect** is a clean, professional clinical dashboard UI built for doctors and healthcare providers. It gives medical staff a unified workspace to manage patient records, track appointments, review vitals, and monitor key clinical metrics — all from a single, intuitive interface.

> Built with React + Vite + Vanilla CSS. No unnecessary dependencies. No bloat. Just clean, production-quality code.

---

## ✨ Features

| Module | Description |
|---|---|
| **Dashboard Overview** | At-a-glance metrics: active patients, today's appointments, pending reports |
| **Patient Directory** | Searchable, filterable data table with patient roster |
| **Patient Details** | Per-patient view with demographics, vitals, and a medical history timeline |
| **Appointment Scheduler** | Day/week calendar view for scheduling and reviewing visits |
| **Clinical Navigation** | Persistent sidebar with quick access to all modules |
| **Smart Topbar** | Global search, notification badge, and doctor profile display |

---

## 🗂️ Project Structure

```
doctor-ui/
├── public/
│   └── favicon.svg
├── src/
│   ├── components/
│   │   ├── Sidebar.jsx         # Primary navigation sidebar
│   │   ├── Sidebar.css
│   │   ├── Topbar.jsx          # Search bar & user profile header
│   │   ├── Topbar.css
│   │   ├── Layout.jsx          # Wrapper that composes Sidebar + Topbar
│   │   ├── Dashboard.jsx       # Overview metrics and quick stats
│   │   ├── Dashboard.css
│   │   ├── PatientList.jsx     # Patient directory table with search
│   │   ├── PatientList.css
│   │   ├── PatientDetails.jsx  # Individual patient record + timeline
│   │   ├── PatientDetails.css
│   │   ├── Appointments.jsx    # Schedule calendar view
│   │   └── Appointments.css
│   ├── App.jsx                 # Root application and routing logic
│   ├── App.css
│   ├── index.css               # Global design tokens and base styles
│   └── main.jsx
├── index.html
├── vite.config.js
└── package.json
```

---

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) v18 or higher
- npm v9 or higher

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Gaman-123/Spot-Coders_backend.git

# 2. Navigate to the frontend directory
cd Spot-Coders_backend/doctor-ui

# 3. Install dependencies
npm install

# 4. Start the development server
npm run dev
```

The application will be available at **http://localhost:5173**

### Build for Production

```bash
npm run build
```

---

## 🎨 Design System

The UI is built around a carefully curated design system defined in `index.css`:

| Token | Value | Usage |
|---|---|---|
| `--primary-color` | `#0052CC` | Primary actions, active states |
| `--secondary-color` | `#00B8D9` | Secondary highlights |
| `--success` | `#36B37E` | Stable status, positive trends |
| `--warning` | `#FFAB00` | Review status, pending items |
| `--danger` | `#FF5630` | Critical alerts, destructive actions |
| `--bg-main` | `#F4F5F7` | Page background |
| `--bg-surface` | `#FFFFFF` | Card & panel surfaces |

**Typography:** System font stack with Inter as the preferred face.  
**Icons:** [Lucide React](https://lucide.dev/) — consistent, lightweight SVG icons.

---

## 🛠️ Development Phases

This frontend was built incrementally across 8 structured phases:

| Phase | Focus | Commit |
|---|---|---|
| 1 | Project scaffold & global CSS tokens | `Initialize doctor interface React application structure` |
| 2 | Sidebar + Topbar core layout | `Implement core navigation layout and routing structure` |
| 3 | Dashboard metrics overview | `Add dashboard overview with key medical metrics` |
| 4 | Patient directory table | `Create patient directory and data table view` |
| 5 | Patient details & medical history | `Develop detailed patient medical record view` |
| 6 | Appointment scheduler calendar | `Implement appointment scheduling interface` |
| 7 | UI polish, animations, responsiveness | `Enhance user interface styling and responsiveness` |
| 8 | Documentation | `Update repository documentation and project instructions` |

---

## 🤝 Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes with a clear, descriptive message
4. Open a Pull Request describing what you changed and why

Please keep commit messages professional and descriptive.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with care for the people who care for us.

</div>
