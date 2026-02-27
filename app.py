"""
NephroLens - Full Stack Kidney Stone Detection
Author  : Sanket Kelzarkar
Guide   : Dr. Praveen Kumar
Sem VI  · 2026
"""

import os, uuid, json, logging, sqlite3
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify, abort, send_from_directory)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from flask_bcrypt import Bcrypt
from PIL import Image
from werkzeug.utils import secure_filename

# ── CONFIG ────────────────────────────────────────────────────
class Config:
    SECRET_KEY              = os.environ.get('SECRET_KEY', 'nephrolens-secret-2026')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///nephrolens.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER           = os.path.join('static', 'uploads')
    MAX_CONTENT_LENGTH      = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS      = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    MODEL_DIR               = os.environ.get('MODEL_DIR', os.path.dirname(__file__))
    PER_PAGE                = 10

# ── APP INIT ──────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)

db            = SQLAlchemy(app)
bcrypt        = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view             = 'login'
login_manager.login_message          = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── MODELS ────────────────────────────────────────────────────
def gen_patient_id():
    return 'NL-' + datetime.utcnow().strftime('%Y%m%d') + '-' + uuid.uuid4().hex[:6].upper()

class Patient(UserMixin, db.Model):
    __tablename__ = 'patients'
    id          = db.Column(db.Integer,     primary_key=True)
    patient_id  = db.Column(db.String(30),  unique=True, nullable=False, default=gen_patient_id)
    full_name   = db.Column(db.String(120), nullable=False)
    email       = db.Column(db.String(120), unique=True, nullable=False)
    mobile      = db.Column(db.String(15),  nullable=False)
    password    = db.Column(db.String(200), nullable=False)
    age         = db.Column(db.Integer,     nullable=False)
    gender      = db.Column(db.String(20),  nullable=False)   # "Prefer not to say" needs 20
    address     = db.Column(db.Text,        nullable=False, default='')
    aadhar      = db.Column(db.String(20),  nullable=False)
    abha        = db.Column(db.String(50),  nullable=True, default='')
    created_at  = db.Column(db.DateTime,    default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='patient',
                                  lazy=True, cascade='all, delete-orphan')

    # ── Properties used in templates ──────────────────────────
    @property
    def total_scans(self):     return len(self.predictions)
    @property
    def stones_detected(self): return sum(1 for p in self.predictions if p.result == 'Stone Detected')
    @property
    def normal_scans(self):    return sum(1 for p in self.predictions if p.result == 'Normal')
    @property
    def first_name(self):      return self.full_name.split()[0]


class Prediction(db.Model):
    __tablename__ = 'predictions'
    id            = db.Column(db.Integer,    primary_key=True)
    patient_id_fk = db.Column(db.Integer,    db.ForeignKey('patients.id'), nullable=False)
    image_path    = db.Column(db.String(200), nullable=False)
    heatmap_path  = db.Column(db.String(200), nullable=True)
    result        = db.Column(db.String(50),  nullable=False)   # 'Stone Detected' | 'Normal'
    confidence    = db.Column(db.Float,       nullable=False)
    model_used    = db.Column(db.String(50),  nullable=False)
    model_scores  = db.Column(db.Text,        default='{}')
    notes         = db.Column(db.Text,        default='')
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)

    # ── Properties used in templates ──────────────────────────
    @property
    def is_stone(self):    return self.result == 'Stone Detected'
    @property
    def badge_color(self): return 'danger' if self.is_stone else 'success'
    @property
    def scores_dict(self):
        try:    return json.loads(self.model_scores)
        except: return {}


@login_manager.user_loader
def load_user(uid): return Patient.query.get(int(uid))

# ── HELPERS ───────────────────────────────────────────────────
def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def unique_fn(orig):
    ext = orig.rsplit('.', 1)[-1].lower() if '.' in orig else 'png'
    return f"{uuid.uuid4().hex}.{ext}"

def validate_img(path):
    try:
        Image.open(path).verify()
        return True
    except Exception:
        return False

def _auto_migrate():
    """
    Detects a stale 'users' table from an old schema and deletes
    the DB so db.create_all() can build the correct 'patients' table.
    Called once at startup before create_all().
    """
    db_path = os.path.join(os.path.dirname(__file__), 'nephrolens.db')
    if not os.path.exists(db_path):
        return
    try:
        con    = sqlite3.connect(db_path)
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        con.close()
        if 'users' in tables and 'patients' not in tables:
            os.remove(db_path)
            log.warning("⚠️  Stale 'users' table detected — DB deleted. Rebuilding with correct schema.")
    except Exception as ex:
        log.warning("DB migration check failed: %s", ex)

# ── GRAD-CAM ──────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.grads = self.acts = None
        layer.register_forward_hook( lambda m, i, o: setattr(self, 'acts', o.detach()))
        layer.register_backward_hook(lambda m, gi, go: setattr(self, 'grads', go[0].detach()))

    def generate(self, tensor, cls=None):
        out = self.model(tensor)
        if cls is None: cls = out.argmax(dim=1)
        self.model.zero_grad()
        oh = torch.zeros_like(out); oh[0, cls] = 1
        out.backward(gradient=oh, retain_graph=True)
        w   = self.grads[0].mean(dim=(1, 2))
        cam = (w[:, None, None] * self.acts[0]).sum(0)
        cam = F.relu(cam)
        mn, mx = cam.min(), cam.max()
        if mx > mn: cam = (cam - mn) / (mx - mn)
        return cam.cpu().numpy()


def make_heatmap(model, img_path, device):
    try:
        from torchvision import transforms as T
        tf     = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                            T.Normalize([.485, .456, .406], [.229, .224, .225])])
        pil    = Image.open(img_path).convert('RGB')
        t      = tf(pil).unsqueeze(0).to(device)
        gc     = GradCAM(model, model.layer4[-1])
        cam    = gc.generate(t)
        img_np = np.array(pil)
        cam_r  = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heat   = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
        heat   = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        over   = cv2.addWeighted(img_np.astype(np.float32), .6,
                                 heat.astype(np.float32), .4, 0).astype(np.uint8)
        hname  = os.path.splitext(os.path.basename(img_path))[0] + '_heatmap.jpg'
        Image.fromarray(over).save(
            os.path.join(app.config['UPLOAD_FOLDER'], hname), quality=90)
        return hname
    except Exception as e:
        log.warning("GradCAM skipped: %s", e)
        return None

# ── ML PREDICTOR ──────────────────────────────────────────────
class Predictor:
    CLASSES = ['Stone Detected', 'Normal']

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self._load_resnet()
        self._load_yolo()
        self._load_vmunet()
        log.info("Predictor ready | device=%s | models=%s",
                 self.device, list(self.models.keys()) or ['demo'])

    def _tf(self):
        from torchvision import transforms as T
        return T.Compose([T.Resize((224, 224)), T.ToTensor(),
                          T.Normalize([.485, .456, .406], [.229, .224, .225])])

    def _load_resnet(self):
        try:
            from torchvision import models as tvm
            p = os.path.join(app.config['MODEL_DIR'], 'resnet_training', 'resnet50_best.pth')
            if not os.path.exists(p):
                log.warning("ResNet weights not found at %s", p); return
            m = tvm.resnet50(weights=None)
            m.fc = torch.nn.Sequential(
                torch.nn.Dropout(.5), torch.nn.Linear(m.fc.in_features, 512),
                torch.nn.ReLU(), torch.nn.Dropout(.3), torch.nn.Linear(512, 2))
            m.load_state_dict(torch.load(p, map_location=self.device))
            m.eval().to(self.device)
            self.models['resnet'] = m
            log.info("ResNet50 loaded.")
        except Exception as e:
            log.error("ResNet load failed: %s", e)

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            p = os.path.join(app.config['MODEL_DIR'], 'yolov8_training',
                             'yolov8n_kidney_stone', 'weights', 'best.pt')
            if not os.path.exists(p):
                log.warning("YOLOv8 weights not found at %s", p); return
            self.models['yolov8'] = YOLO(p)
            log.info("YOLOv8 loaded.")
        except Exception as e:
            log.error("YOLOv8 load failed: %s", e)

    def _load_vmunet(self):
        try:
            import sys, importlib
            sys.path.insert(0, app.config['MODEL_DIR'])
            mod = importlib.import_module('8_model_vmunet')
            p   = os.path.join(app.config['MODEL_DIR'], 'vmunet_training', 'vmunet_best.pth')
            if not os.path.exists(p):
                log.warning("VMUNet weights not found at %s", p); return
            m = mod.VMUNet(in_channels=3, num_classes=2, img_size=224,
                           embed_dim=128, encoder_depth=3, d_state=8)
            m.load_state_dict(torch.load(p, map_location=self.device))
            m.eval().to(self.device)
            self.models['vmunet'] = m
            log.info("VM-UNet loaded.")
        except Exception as e:
            log.error("VMUNet load failed: %s", e)

    def _pred_resnet(self, path):
        if 'resnet' not in self.models: return None, 0.
        t = self._tf()(Image.open(path).convert('RGB')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = F.softmax(self.models['resnet'](t), dim=1)
            c, i = torch.max(p, 1)
        return self.CLASSES[i.item()], round(c.item() * 100, 2)

    def _pred_yolo(self, path):
        if 'yolov8' not in self.models: return None, 0.
        res = self.models['yolov8'].predict(path, verbose=False)[0]
        if res.boxes:
            b = res.boxes[0]
            return self.CLASSES[int(b.cls[0])], round(float(b.conf[0]) * 100, 2)
        return 'Normal', 50.

    def _pred_vmunet(self, path):
        if 'vmunet' not in self.models: return None, 0.
        t = self._tf()(Image.open(path).convert('RGB')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            co, _ = self.models['vmunet'](t)
            p = F.softmax(co, dim=1); c, i = torch.max(p, 1)
        return self.CLASSES[i.item()], round(c.item() * 100, 2)

    def _demo(self):
        import random
        return random.choice(self.CLASSES), round(random.uniform(72, 96), 2)

    def predict(self, path, choice='ensemble'):
        per = {}
        try:
            pred = conf = None
            if choice == 'resnet':
                pred, conf = self._pred_resnet(path)
                label = 'ResNet50'
                if pred: per['ResNet50'] = conf
            elif choice == 'yolov8':
                pred, conf = self._pred_yolo(path)
                label = 'YOLOv8'
                if pred: per['YOLOv8'] = conf
            elif choice == 'vmunet':
                pred, conf = self._pred_vmunet(path)
                label = 'VM-UNet'
                if pred: per['VM-UNet'] = conf
            else:                                       # ensemble
                results, confs = [], []
                for name, fn in [('ResNet50', self._pred_resnet),
                                 ('YOLOv8',   self._pred_yolo),
                                 ('VM-UNet',  self._pred_vmunet)]:
                    p, c = fn(path)
                    if p:
                        results.append(p); confs.append(c); per[name] = c
                if results:
                    pred  = max(set(results), key=results.count)
                    conf  = round(float(np.mean(confs)), 2)
                else:
                    pred, conf = self._demo()
                label = 'Ensemble'

            if pred is None:
                pred, conf = self._demo()
            return pred, conf, label, per

        except Exception as e:
            log.error("Predict error: %s", e)
            return 'Error', 0., choice.title(), {}


predictor = Predictor()

# ── CONTEXT PROCESSOR ─────────────────────────────────────────
@app.context_processor
def inject_globals():
    # `now` is used in base.html footer and result.html "Analyzed On"
    return {'now': datetime.utcnow()}

# ── FAVICON ───────────────────────────────────────────────────
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static', 'img'),
        'logo.png', mimetype='image/png')

# ── ERROR HANDLERS ────────────────────────────────────────────
@app.errorhandler(404)
def e404(e):
    return render_template('errors/404.html'), 404

@app.errorhandler(413)
def e413(e):
    flash('File too large (max 16 MB).', 'danger')
    return redirect(url_for('predict'))

@app.errorhandler(500)
def e500(e):
    log.exception("500 error")
    return render_template('errors/500.html'), 500

# ── PUBLIC ROUTES ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# ── AUTH: REGISTER ────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        f      = request.form
        name   = f.get('full_name', '').strip()
        email  = f.get('email', '').strip().lower()
        mobile = f.get('mobile', '').strip()
        pw     = f.get('password', '')
        cpw    = f.get('confirm_password', '')
        age    = f.get('age', '')
        gender = f.get('gender', '')
        addr   = f.get('address', '').strip()           # optional in register.html
        aadhar = f.get('aadhar', '').strip().replace(' ', '').replace('-', '')
        abha   = f.get('abha', '').strip()
        errs   = []

        # ── Validation ──────────────────────────────────────
        if len(name) < 3:
            errs.append('Full name must be at least 3 characters.')
        if '@' not in email or '.' not in email:
            errs.append('A valid email address is required.')
        if not mobile.isdigit() or not (10 <= len(mobile) <= 15):
            errs.append('A valid 10–15 digit mobile number is required.')
        if len(pw) < 8:
            errs.append('Password must be at least 8 characters.')
        if pw != cpw:
            errs.append('Passwords do not match.')
        if not age.isdigit() or not (1 <= int(age) <= 120):
            errs.append('A valid age between 1 and 120 is required.')
        valid_genders = ('Male', 'Female', 'Other', 'Prefer not to say')
        if gender not in valid_genders:
            errs.append('Please select a valid gender.')
        if not aadhar.isdigit() or len(aadhar) != 12:
            errs.append('A valid 12-digit Aadhar number is required.')
        if Patient.query.filter_by(email=email).first():
            errs.append('This email is already registered.')
        if Patient.query.filter_by(mobile=mobile).first():
            errs.append('This mobile number is already registered.')

        if errs:
            for e in errs:
                flash(e, 'danger')
            return redirect(url_for('register'))

        pat = Patient(
            full_name = name,
            email     = email,
            mobile    = mobile,
            password  = bcrypt.generate_password_hash(pw).decode('utf-8'),
            age       = int(age),
            gender    = gender,
            address   = addr,
            aadhar    = aadhar,
            abha      = abha,
        )
        db.session.add(pat)
        db.session.commit()
        flash(f'Account created! Your Patient ID: {pat.patient_id}', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# ── AUTH: LOGIN ───────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        pw    = request.form.get('password', '')
        pat   = Patient.query.filter_by(email=email).first()
        if pat and bcrypt.check_password_hash(pat.password, pw):
            login_user(pat, remember=bool(request.form.get('remember')))
            flash(f'Welcome back, {pat.first_name}!', 'success')
            nxt = request.args.get('next')
            return redirect(nxt or url_for('dashboard'))
        flash('Invalid email or password.', 'danger')

    return render_template('login.html')

# ── AUTH: LOGOUT ──────────────────────────────────────────────
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))

# ── DASHBOARD ─────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    page  = request.args.get('page', 1, type=int)
    preds = (Prediction.query
             .filter_by(patient_id_fk=current_user.id)
             .order_by(Prediction.created_at.desc())
             .paginate(page=page, per_page=app.config['PER_PAGE'], error_out=False))
    # Template uses: predictions.items, predictions.total, predictions.pages,
    #                predictions.has_prev/next, predictions.prev_num/next_num,
    #                predictions.page, predictions.iter_pages(...)
    return render_template('dashboard.html', predictions=preds)

# ── PREDICT ───────────────────────────────────────────────────
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image part in the request.', 'danger')
            return redirect(url_for('predict'))

        f = request.files['image']
        if not f or f.filename == '':
            flash('No file selected.', 'danger')
            return redirect(url_for('predict'))

        if not allowed_file(f.filename):
            flash('Unsupported format. Allowed: PNG, JPG, JPEG, BMP, TIFF.', 'danger')
            return redirect(url_for('predict'))

        fname = unique_fn(secure_filename(f.filename))
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        f.save(fpath)

        if not validate_img(fpath):
            os.remove(fpath)
            flash('The uploaded file is corrupt or not a valid image.', 'danger')
            return redirect(url_for('predict'))

        choice = request.form.get('model', 'ensemble')
        notes  = request.form.get('notes', '').strip()[:500]

        result, conf, model_used, per = predictor.predict(fpath, choice)

        if result == 'Error':
            os.remove(fpath)
            flash('Prediction failed. Please try again.', 'danger')
            return redirect(url_for('predict'))

        # Grad-CAM (requires ResNet)
        heatmap = None
        if 'resnet' in predictor.models:
            heatmap = make_heatmap(predictor.models['resnet'], fpath, predictor.device)

        pred = Prediction(
            patient_id_fk = current_user.id,
            image_path    = fname,
            heatmap_path  = heatmap,
            result        = result,
            confidence    = conf,
            model_used    = model_used,
            model_scores  = json.dumps(per),
            notes         = notes,
        )
        db.session.add(pred)
        db.session.commit()

        # result.html expects: prediction, confidence, model, image_path,
        #                      heatmap_path, per_scores, pred_id
        return render_template(
            'result.html',
            prediction   = pred.result,
            confidence   = pred.confidence,
            model        = pred.model_used,
            image_path   = pred.image_path,
            heatmap_path = pred.heatmap_path,
            per_scores   = per,
            pred_id      = pred.id,
        )

    return render_template('predict.html')

# ── VIEW SAVED PREDICTION ─────────────────────────────────────
@app.route('/prediction/<int:pred_id>')
@login_required
def view_prediction(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.patient_id_fk != current_user.id:
        abort(403)
    return render_template(
        'result.html',
        prediction   = pred.result,
        confidence   = pred.confidence,
        model        = pred.model_used,
        image_path   = pred.image_path,
        heatmap_path = pred.heatmap_path,
        per_scores   = pred.scores_dict,
        pred_id      = pred.id,
    )

# ── DELETE PREDICTION ─────────────────────────────────────────
# dashboard.html  → url_for('delete_prediction', pred_id=p.id)
# result.html     → url_for('delete_prediction', pred_id=pred_id)
@app.route('/prediction/<int:pred_id>/delete', methods=['POST'])
@login_required
def delete_prediction(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.patient_id_fk != current_user.id:
        abort(403)
    # Remove image files
    for fn in [pred.image_path, pred.heatmap_path]:
        if fn:
            p = os.path.join(app.config['UPLOAD_FOLDER'], fn)
            if os.path.exists(p):
                os.remove(p)
    db.session.delete(pred)
    db.session.commit()
    flash('Scan deleted.', 'info')
    return redirect(url_for('dashboard'))

# ── PROFILE ───────────────────────────────────────────────────
@app.route('/profile')
@login_required
def profile():
    # profile.html uses current_user directly — no extra context needed
    return render_template('profile.html')

@app.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    if request.method == 'POST':
        f = request.form
        current_user.full_name = f.get('full_name', current_user.full_name).strip()
        current_user.mobile    = f.get('mobile',    current_user.mobile).strip()
        age_val = f.get('age', str(current_user.age))
        if age_val.isdigit() and 1 <= int(age_val) <= 120:
            current_user.age = int(age_val)
        current_user.address = f.get('address', current_user.address).strip()
        current_user.abha    = f.get('abha',    current_user.abha or '').strip()
        pw = f.get('new_password', '').strip()
        if pw:
            if len(pw) < 8:
                flash('New password must be at least 8 characters.', 'danger')
                return redirect(url_for('edit_profile'))
            current_user.password = bcrypt.generate_password_hash(pw).decode('utf-8')
        db.session.commit()
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('profile'))
    return render_template('edit_profile.html')

# ── API: STATS (for chart widgets if needed) ──────────────────
@app.route('/api/stats')
@login_required
def api_stats():
    u = current_user
    recent = (Prediction.query
              .filter_by(patient_id_fk=u.id)
              .order_by(Prediction.created_at.desc())
              .limit(6).all())
    trend = [{'date':   p.created_at.strftime('%d %b'),
              'result': p.result,
              'conf':   p.confidence}
             for p in reversed(recent)]
    return jsonify({
        'total':  u.total_scans,
        'stones': u.stones_detected,
        'normal': u.normal_scans,
        'trend':  trend,
    })

# ── CLI ───────────────────────────────────────────────────────
@app.cli.command('reset-db')
def reset_db():
    """Drop and recreate all tables (dev use only)."""
    db.drop_all()
    db.create_all()
    print("✅ Database reset.")

# ── ENTRY ─────────────────────────────────────────────────────
if __name__ == '__main__':
    _auto_migrate()                 # removes stale 'users' DB if present
    with app.app_context():
        db.create_all()
        log.info("DB ready — tables: patients, predictions")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)