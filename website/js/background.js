class LogEntry {
    constructor(x, index) {
        this.x = x;
        this.y = window.innerHeight / 2;
        this.index = index;
        
        // Randomize sizes for Key and Value to simulate real data
        // Header is constant 15 units (representing 15 bytes)
        this.headerWidth = 40; 
        this.keyWidth = 60 + Math.random() * 80;
        this.valueWidth = 100 + Math.random() * 150;
        
        this.width = this.headerWidth + this.keyWidth + this.valueWidth;
        this.height = 80;
        
        this.opacity = 0;
        this.targetOpacity = 0.6;
        
        // Colors - darker for lighter backgrounds
        this.colorHeader = 'rgba(40, 40, 50, 1)';
        this.colorKey = 'rgba(0, 160, 130, 1)'; // Darker Teal
        this.colorValue = 'rgba(50, 110, 180, 1)'; // Darker Blue
    }

    update(speed) {
        this.x -= speed;
        if (this.opacity < this.targetOpacity) {
            this.opacity += 0.02;
        }
    }

    draw(ctx) {
        if (this.x + this.width < 0 || this.x > window.innerWidth) return; // Cull off-screen

        const y = this.y - this.height / 2;
        
        ctx.lineWidth = 1;
        ctx.lineJoin = 'round';
        
        // Header
        this.drawSegment(ctx, this.x, y, this.headerWidth, this.colorHeader, 'HDR');
        
        // Key
        this.drawSegment(ctx, this.x + this.headerWidth, y, this.keyWidth, this.colorKey, 'KEY');
        
        // Value
        this.drawSegment(ctx, this.x + this.headerWidth + this.keyWidth, y, this.valueWidth, this.colorValue, 'VAL');

        // Connector line
        ctx.beginPath();
        ctx.moveTo(this.x, y + this.height + 10);
        ctx.lineTo(this.x + this.width, y + this.height + 10);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
        ctx.stroke();
        
        // Index Label below
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.font = '10px "JetBrains Mono"';
        ctx.fillText(`Entry ${this.index}`, this.x, y + this.height + 25);
    }

    drawSegment(ctx, x, y, w, color, label) {
        const h = this.height;
        
        // Glow
        ctx.shadowBlur = 15;
        ctx.shadowColor = color;
        
        // Border
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        
        // Slight fill - more opaque for visibility
        ctx.fillStyle = color.replace('1)', '0.15)');
        ctx.fillRect(x, y, w, h);
        
        // Reset Glow for text
        ctx.shadowBlur = 0;
        
        // Label inside
        ctx.fillStyle = color;
        ctx.font = 'bold 10px Inter';
        ctx.fillText(label, x + 10, y + 20);
        
        // Byte size simulation
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.font = '9px "JetBrains Mono"';
        ctx.fillText(`${Math.floor(w/2)}b`, x + 10, y + h - 10);
    }
}

class LogSimulation {
    constructor() {
        this.canvas = document.getElementById('log-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.entries = [];
        this.speed = 0.5;
        this.lastIndex = 0;
        
        this.resize();
        window.addEventListener('resize', () => this.resize());
        
        // Fill initial screen
        let currentX = 50;
        while(currentX < window.innerWidth) {
            const entry = new LogEntry(currentX, this.lastIndex++);
            entry.opacity = entry.targetOpacity; // Initial ones fully visible
            this.entries.push(entry);
            currentX += entry.width + 20; // 20px gap
        }
        
        this.animate();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Update and replace
        const tail = this.entries[this.entries.length - 1];
        if (tail.x + tail.width < window.innerWidth + 200) {
            const newEntry = new LogEntry(tail.x + tail.width + 20, this.lastIndex++);
            this.entries.push(newEntry);
        }
        
        // Filter out off-screen
        if (this.entries[0].x + this.entries[0].width < -100) {
            this.entries.shift();
        }

        // Draw / Update
        this.entries.forEach(entry => {
            entry.update(this.speed);
            entry.draw(this.ctx);
        });

        // Loop
        requestAnimationFrame(() => this.animate());
    }
}

// Init when DOM loaded
document.addEventListener('DOMContentLoaded', () => {
    new LogSimulation();
});
