#!/usr/bin/env python3
"""Written with the help of Claude 3.5 Sonnet"""
import sys
import json
import os
from pathlib import Path
import hashlib
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QMessageBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt

# Minimal HTML template
TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <script defer src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>body{margin:0;padding:0;}</style>
</head>
<body>
    <div id="plot"></div>
    <script>
        window.addEventListener('load', function() {
            Plotly.newPlot('plot', %s, %s, {responsive: true});
        });
    </script>
</body>
</html>
'''

class PlotlyViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plotly Viewer")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Create web view
        self.web_view = QWebEngineView()
        self.layout.addWidget(self.web_view)
        
        # Connect signal for window resizing
        self.web_view.loadFinished.connect(self.adjust_window_size)
        
        # Initialize cache directory
        self.cache_dir = Path.home() / '.plotly_viewer_cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        self.fig_width = None
        self.fig_height = None
        
        self.zoom_factor = 1.0
        
    def get_cache_path(self, content):
        """Generate a cache file path based on content hash"""
        content_hash = hashlib.md5(str(content).encode()).hexdigest()
        return self.cache_dir / f"{content_hash}.html"

    def create_html(self, fig_dict):
        """Create HTML content from figure dictionary"""
        data = fig_dict.get('data', [])
        layout = fig_dict.get('layout', {})
        
        if 'margin' not in layout:
            layout['margin'] = dict(l=10, r=10, t=30, b=10)
        
        return TEMPLATE % (json.dumps(data), json.dumps(layout))

    def adjust_window_size(self, ok):
        if ok:
            if self.fig_width is not None and self.fig_height is not None:
                self.resize_window([self.fig_width, self.fig_height])
            else:
                js = """
                var rect = document.querySelector('.plotly').getBoundingClientRect();
                [rect.width, rect.height];
                """
                self.web_view.page().runJavaScript(js, self.resize_window)

    def resize_window(self, dimensions):
        if dimensions:
            width, height = dimensions
            width += 40
            height += 80
            
            cursor_pos = QApplication.desktop().cursor().pos()
            screen = QApplication.desktop().screenGeometry(cursor_pos)
            
            MIN_WIDTH, MIN_HEIGHT = 400, 300
            max_width = int(screen.width() * 0.9)
            max_height = int(screen.height() * 0.9)
            
            width = max(MIN_WIDTH, min(width, max_width))
            height = max(MIN_HEIGHT, min(height, max_height))
            
            x = screen.x() + (screen.width() - width) // 2
            y = screen.y() + (screen.height() - height) // 2
            
            self.setGeometry(x, y, width, height)
            if not self.isVisible():
                self.show()

    def load_plot(self, filename):
        try:
            with open(filename, 'r') as f:
                fig_dict = json.load(f)
            
            try:
                layout = fig_dict.get('layout', {})
                self.fig_width = layout.get('width')
                self.fig_height = layout.get('height')
            except:
                self.fig_width = None
                self.fig_height = None
            
            cache_path = self.get_cache_path(fig_dict)
            
            if not cache_path.exists():
                html_content = self.create_html(fig_dict)
                with open(cache_path, 'w') as f:
                    f.write(html_content)
            
            url = QUrl.fromLocalFile(str(cache_path.absolute()))
            self.web_view.setUrl(url)
            
            self.setWindowTitle(f"Plotly Viewer - {os.path.basename(filename)}")
            
        except json.JSONDecodeError:
            QMessageBox.critical(self, "Error", "Invalid JSON file")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading plot: {str(e)}")
    
    # Zoom in/out with ctrl++/ctrl+-
    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
                self.zoom_factor *= 1.25
                self.web_view.setZoomFactor(self.zoom_factor)
            elif event.key() == Qt.Key_Minus:
                self.zoom_factor *= 0.8
                self.web_view.setZoomFactor(self.zoom_factor)
            elif event.key() == Qt.Key_0:
                self.zoom_factor = 1.0
                self.web_view.setZoomFactor(self.zoom_factor)
                
def main():
    app = QApplication(sys.argv)
    viewer = PlotlyViewer()
    
    if len(sys.argv) > 1:
        viewer.load_plot(sys.argv[1])
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
