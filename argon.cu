#include <bits/stdc++.h>
#include <cuda.h>

/*===========================================================================*
 *                  	definy aliasy makra i tp		             *
 *===========================================================================*/

using namespace std;
using namespace std::chrono;

using real_t = double;
using ll = long long;
using vr = vector<real_t>;

#define pb push_back
#define tos to_string
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a ## b
#define _ CONCAT(_, __COUNTER__)
#define sq(x) ((x)*(x))
#define def_min(a, b) (a < b ? a : b)

/* pomiary czasu */
auto __my_time_start = steady_clock::now(), __my_time_stop = steady_clock::now();
double get_duration() { return duration<double, milli>(__my_time_stop - __my_time_start).count();}
void start() { __my_time_start = steady_clock::now(); }
void stop() { __my_time_stop = steady_clock::now(); }

/* pliki logów */
ofstream argon("argon.out");
ofstream argon_energia("argon_energia.csv");
ofstream argon_trajektoria("argon_trajektoria.xyz");

/* macro cudaMaloców i tp */
cudaError_t cuda_err;
#define cudaAssert(status) if((cuda_err = status) != cudaSuccess) { \
    cout << __LINE__ << " " << cudaGetErrorString(cuda_err) << endl; exit(1); }

/* maxymalne zagęszczenie atomów */
constexpr int max_zageszczenie(real_t r) {
    return int(24.0*3.14*1.41/15.0 * (r*r*r) / (0.37*0.37*0.37) + 10);
}

/*===========================================================================*
 *                   	        konfiguracja				     *
 *===========================================================================*/

/* stałe z treści */
constexpr real_t sig =	0.369;      /* nm */
constexpr real_t eps =	1.19;	    /* kJ/mol */
constexpr real_t a   =	0.260922;   /* sig = sqrt(2)*a */
constexpr real_t ts  =  0.001;      /* time stamp w symulcji */
constexpr real_t B   =  1;          /* siła balonu */
constexpr real_t mas = 39.95;       /* masa atomu */
constexpr real_t kB  = 0.00831;     /* stała Boltzmana */

/* wymiary siatki z atomami */
constexpr int siat_X = 6;
constexpr int siat_Y = 6;
constexpr int siat_Z = 6;
constexpr int N = siat_X*siat_Y*siat_Z*4; /*ilość atomów */

/* wymagana konfiguracja */
constexpr int DEBUG         = 1;        /* 0-nic, 1-najważniesze, 2-rozszerzone, 3-wszystko */
constexpr bool VERBOSE      = true;     /* czy wypisywać najważniejsze logi również na stdout */
constexpr bool STATS        = false;    /* czy obliczac statystyki energi */
constexpr int Nterm         = 10000;    /* ilość kroków fazy termalizacji */
constexpr int Ngrz          = 5000;     /* ilość kroków fazy grzania */
constexpr int Nch           = 5000;     /* ilość kroków fazy chłodzenia */
constexpr int ENERG         = 100;      /* częstotliowść zapisu anergi */
constexpr int TRAJE         = 100;      /* częstotliwość zapisu współrzędnych */
constexpr real_t rB         = 20;       /* r balonu w nm */
constexpr real_t rOd        = 1.3;      /* promień odcięcia */
constexpr real_t rBuff      = 0.3;      /* bufor */
constexpr real_t start_T    = 70;       /* temperatura początkowa */
constexpr real_t cool_T     = 20;       /* temperatura do której chłodzimy */
constexpr real_t heat_T     = 120;      /* temperatura do której ogrzewamy */

/* max długość listy sąsiadów */
constexpr int MAX_LI = def_min(N, max_zageszczenie(rOd + rBuff));

/*===========================================================================*
 *                  		    utility		     		     *
 *===========================================================================*/

string to_s() { return " ";}
template<class T, class... Ts>
string to_s(T arg, Ts &... args) {
    return to_string(arg) + " " + to_s(args...);
}

inline string tos_stab(real_t x, size_t max_len = 26) {
    string res = tos(x);
    return res;

    assert(res.size() <= max_len);
    while (res.size() < max_len)
         res += " ";
    return res;
}

inline void log(const string& co, int lvl) {
    if (VERBOSE and lvl == 1)
        cout << co << "\n";
    if (lvl <= DEBUG)
        argon << co << "\n";
}

/*===========================================================================*
 *                              tablice globalne                             *
 *===========================================================================*/

/* tablice na hosice */
real_t cords[3*N];      /* gdzie jest atom */
real_t V[3*N];          /* prędkości atomu względem osi */
real_t F[3*N];          /* siła względem osi */
real_t Epot[2*N];       /* energia potencjalna Lennarda Jonesa, [0] - r12, [1] - r6 */
real_t Ebal[N];         /* enrgia potencjalna - człon balonu */
real_t Ekin[N];         /* energia kinetyczna atomu */

int G[N][MAX_LI];       /* lista sąsiedztwa atomów */
real_t poz0[N][3];      /* pozycja atomu w momencie budowy listy sąsiedztwa */

/* gpu I i gpu II */
real_t* F_gpu;          /* siła, gpu */
real_t* Epot_gpu;       /* energia potencjalna, gpu */
real_t* cords_gpu;      /* kordynaty, gpu */

/* wersja gpu 3, podział na atomy w roli hosta i goscia */
real_t* F_gpu_host;
real_t* F_gpu_guest;
real_t* Epot_gpu_host;
real_t* Epot_gpu_guest;

/* wersja gpu 4, listy sasiedztwa */
int* G_gpu;
real_t poz0_cpu[N*3];

/*===========================================================================*
 *                              statystyka                                   *
 *===========================================================================*/

real_t Ekin_snaps[Nterm];
real_t Epot_snaps[Nterm];

void up_stats(int nr) {
    if (!STATS)
        return;
    if (nr >= Nterm)
        return;

    Ekin_snaps[nr] = Epot_snaps[nr] = 0;
    
    for (int i = 0; i < N; i++) {
        Ekin_snaps[nr] += Ekin[i];
        Epot_snaps[nr] += Epot[i*2] + Epot[i*2 + 1];
    }
}

inline real_t get_sr(real_t* tab, int SIZ) {
    real_t sr = 0;
    for (int i = 0; i < SIZ; i++)
        sr += tab[i]/SIZ;
    return sr;
}

inline real_t get_var(real_t* tab, int SIZ, real_t sr) {
    real_t var = 0;
    for (int i = 0; i < SIZ; i++)
        var += sq(sr - tab[i])/N;
    return var;
}

inline void log_stats(real_t* tab, int SIZ, string label) {
    real_t sr = get_sr(tab, SIZ);
    real_t var = get_var(tab, SIZ, sr);
    real_t stdev = sqrt(var);

    log(label, 1);
    log("Sr = "+tos(sr), 1);
    log("Var = "+tos(var), 1);
    log("Stdev = "+tos(stdev)+"\n", 1);
}

void stats_out() {
    if (!STATS)
        return;
    real_t all_en[Nterm];
    for (int i = 0; i < Nterm; i++)
        all_en[i] = Ekin_snaps[i] + Epot_snaps[i];

    log_stats(Ekin_snaps, Nterm, "Energia kinetyczna");
    log_stats(Epot_snaps, Nterm, "Energia potencjalna");
    log_stats(all_en, Nterm, "Energia calkowita");
    log("Bezwzgledna roznica poczatkowej od koncowej = "+tos(abs(all_en[0]-all_en[Nterm-1]))+"\n", 1);
}

/*===========================================================================*
 *              implementacja części wspólnej dla wszystkich wersji	     *
 *===========================================================================*/

inline real_t kinetyczna(int i) {
    real_t res = 0;
    for (int k = 0; k < 3; k++)
        res += sq(V[i*3 + k]);
    return res*mas/2.0;
}

inline constexpr real_t temperatura(real_t kin) { return kin/N*2.0/3.0/kB; }

inline real_t cur_temperatura() {
    real_t sum = 0;
    for (size_t i = 0; i < N; i++)
        sum += Ekin[i];
    return temperatura(sum);
}

/* pierwsza podstawowa wersja, wykorzystamy ja we wspolnej czesci do initu atomow */
void up_forces_cpu_1();

inline void up_V() {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            V[i*3 + k] += F[i*3 + k]/mas*ts/2.0;
}

inline void up_cords() {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            cords[i*3 + k] += V[i*3 + k]*ts;
}

inline void up_kin() {
    for (size_t i = 0; i < N; i++)
        Ekin[i] = kinetyczna(i);
}

constexpr inline real_t scale_T(real_t pocz_T, real_t konc_T, real_t kroki = 1) {
    return sqrt(1.0 + (konc_T-pocz_T)/pocz_T/kroki);
}

inline void up_temp(int nr) {
    if (nr < Nterm)
        return;

    real_t mno;

    /* chłodzenie */
    if (nr >= Nterm and nr < Nterm+Nch) {
        mno = scale_T(cur_temperatura(), cool_T, Nch-nr+Nterm);
    }
    /* grzanie */
    else {
        mno = scale_T(cur_temperatura(), heat_T, Ngrz-nr+Nterm+Nch); 
    }

    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            V[i*3 + k] *= mno;
}

inline void up_logi(int nr) {
    if (nr%TRAJE == 0)
        for (int i = 0; i < N; i++)
            argon_trajektoria << (!i ? "{" : ", ")<<"(" << 
                to_s(cords[i*3],cords[i*3 + 1],cords[i*3 + 2])<<
                ")" << (i+1 == N ? "}\n" :"");
    if (nr%ENERG == 0) {
        real_t kin = 0, bal = 0, r6 = 0, r12 = 0;
        for (int i = 0; i < N; i++) {
            kin += Ekin[i];
            r12 += Epot[i*2];
            r6 += Epot[i*2 + 1];
            bal += Ebal[i];
        }

        argon_energia << tos_stab(kin+bal+r6+r12) << ", "
                      << tos_stab(r12+r6+bal) << ", "
                      << tos_stab(r12) << ", "
                      << tos_stab(r6) << ", "
                      << tos_stab(bal) << ", "
                      << tos_stab(kin) << ", "
                      << tos_stab(temperatura(kin)) << "\n";
    }
}

inline void up_ballon() { 
    for (int i = 0; i < N; i++) {
        real_t r = 0;
        for (int k = 0; k < 3; k++)
            r += sq(cords[i*3 + k]);
        r = sqrt(r);

        if (r <= rB)
            return;

        real_t mno = B*(r-rB)/r;
        for (int k = 0; k < 3; k++)
            F[i*3 + k] -= mno*cords[i*3 + k];
        Ebal[i] += B*sq(r-rB)/2.0;
    }
}

void init_atomow() {
    log("Inicjuję atomy", 2);

    real_t X_pom [] = {0.5, 0.5, 1.5, 1.5};
    real_t Y_pom [] = {0.5, 1.5, 0.5, 1.5};
    real_t Z_pom [] = {0.5, 1.5, 1.5, 0.5};

    real_t bound = sqrt(0.00831*start_T*6/mas); /* sqrt(6kT/m) */

    uniform_real_distribution<real_t> unif(-bound, bound);
    default_random_engine rand_eng;

    for (int i = 0; i < siat_X; i++)
        for (int j = 0; j < siat_Y; j++)
            for (int k = 0; k < siat_Z; k++) {
                int l = siat_Y*siat_Z*4*i + siat_Z*j*4 + k*4;
                log("Inicjuję komórkę elementarną o wsp. "+tos(i)+" "+tos(j)+" "+tos(k), 3);
                for (int h = 0; h < 4; h++) {
                    cords[(l+h)*3] = a*(2.0*i + X_pom[h]);
                    cords[(l+h)*3 + 1] = a*(2.0*j + Y_pom[h]);
                    cords[(l+h)*3 + 2] = a*(2.0*k + Z_pom[h]);

                    for (int k = 0; k < 3; k++)
                        V[(l+h)*3 + k] = unif(rand_eng);
                }
            }

    log("Wyliczam początkowe przyśpieszenie atomów", 2);

    up_forces_cpu_1();

    for (int i = 0; i < N; i++)
        Ekin[i] = kinetyczna(i);

    log("Zeruje prędkość środka masy i ustawiam środek masy w (0, 0, 0)", 2);
    real_t sC[3] = {0, 0, 0}, sV[3] = {0, 0, 0};
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++) {
            sC[k] += cords[i*3 + k];
            sV[k] += V[i*3 + k];
        }

    for (int k = 0; k < 3; k++) {
        sC[k] /= N;
        sV[k] /= N;
    }

    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++) {
            cords[i*3 + k] -= sC[k];
            V[i*3 + k] -= sV[k];
        }   

    log("Predkosc srodka masy to "+to_s(sV[0], sV[1], sV[2]), 3);
    log("Srodek masy ukladu to "+to_s(sC[0], sC[1], sC[2]), 3);
    log("Skluje prędkości, by temperatura początkowa była równa "+to_s(start_T), 2);

    real_t mno = scale_T(cur_temperatura(), start_T);
    for (int i = 0; i < N; i++)
         for (int k = 0; k < 3; k++)
             V[i*3 + k] *= mno;

    log("Temperatura wynosi "+to_s(cur_temperatura()), 2);
    log("Koniec inicjacji atomow", 2);
}

/*===========================================================================*
 *                  	aktualizacja sił wersje CPU		     	     *
 *===========================================================================*/

inline static void clear_cpu() {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < 3; k++)
            F[i*3 + k] = 0;
        Epot[i*2] = 0;
        Epot[i*2 + 1] = 0;
    }
}

inline static void post_up_cpu() {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < 3; k++)
            F[i*3 + k] *= 12.0*eps;

        Epot[i*2] *= eps/2.0;
        Epot[i*2 + 1] *= eps;
    }
}

void up_forces_cpu_1() {
    clear_cpu();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j)
                continue;

            real_t rij[3];
            for (int k = 0; k < 3; k++)
                rij[k] = cords[i*3 + k] - cords[j*3 + k];
        
            real_t dl_rij = 0;
            for (int k = 0; k < 3; k++)
                dl_rij += sq(rij[k]);
            if (rOd > 0.0 and dl_rij > sq(rOd))
                continue;

            real_t sig2 = sig*sig/dl_rij;
            real_t sig6 = sig2*sig2*sig2;
            real_t sig12 = sig6*sig6;

            real_t dif = (sig12 - sig6)/dl_rij;
        
            for (int k = 0; k < 3; k++)
                F[i*3 + k] += dif*rij[k];

            Epot[i*2] += sig12;
            Epot[i*2 + 1] -= sig6;
        }
    }

    post_up_cpu();
}

/* dodajemy opt: 3 zasada newtona */
void up_forces_cpu_2() {
    clear_cpu();

    for (int i = 0; i < N; i++)
        for (int j = i+1; j < N; j++) {
            real_t rij[3];
            for (int k = 0; k < 3; k++)
                rij[k] = cords[i*3 + k] - cords[j*3 + k];
        
            real_t dl_rij = 0;
            for (int k = 0; k < 3; k++)
                dl_rij += sq(rij[k]);

            if (rOd > 0.0 and dl_rij > sq(rOd))
                continue;

            real_t sig2 = sig*sig/dl_rij;
            real_t sig6 = sig2*sig2*sig2;
            real_t sig12 = sig6*sig6;

            real_t dif = (sig12 - sig6)/dl_rij;
        
            for (int k = 0; k < 3; k++) {
                F[i*3 + k] += dif*rij[k];
                F[j*3 + k] -= dif*rij[k];
            }

            Epot[i*2] += sig12;
            Epot[i*2 + 1] -= sig6;
            Epot[j*2] += sig12;
            Epot[j*2 + 1] -= sig6;
        }

    post_up_cpu();
}

/* dodajemy opt: listy sąsiedztwa */
void rebuild_G_cpu() {
    int pom[N];
    fill(pom, pom+N, 0);

    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            real_t r = 0;
            for (int k = 0; k < 3; k++)
                r += sq(cords[i*3 + k]-cords[j*3 + k]);
            if (r <= sq(rOd + rBuff))
                G[i][pom[i]++] = j;
        }
    }

    for (int i = 0; i < N; i++) {
        G[i][pom[i]] = -1;
        assert(pom[i] < MAX_LI);
        for (int k = 0; k < 3; k++)
            poz0[i][k] = cords[i*3 + k];
    }
}

inline void check_G() {
    static bool first_builded = false;
    if (first_builded == false) {
        rebuild_G_cpu();
        first_builded = true;
        return;
    }

    for (int i = 0; i < N; i++) {
        real_t dl = 0;
        for (int k = 0; k < 3; k++)
            dl += sq(poz0[i][k]-cords[i*3 + k]);
        if (dl*4.0 >= rBuff*rBuff) {
            rebuild_G_cpu();
            return;
        }
    }
}

void up_forces_cpu_3() {
    check_G();
    clear_cpu();

    for (int i = 0; i < N; i++)
        for (int pm = 0, j = G[i][pm]; G[i][pm] != -1; j = G[i][++pm]) {
            real_t rij[3];
            for (int k = 0; k < 3; k++)
                rij[k] = cords[i*3 + k] - cords[j*3 + k];
        
            real_t dl_rij = 0;
            for (int k = 0; k < 3; k++)
                dl_rij += sq(rij[k]);

            if (rOd > 0.0 and dl_rij > sq(rOd))
                continue;

            real_t sig2 = sig*sig/dl_rij;
            real_t sig6 = sig2*sig2*sig2;
            real_t sig12 = sig6*sig6;

            real_t dif = (sig12 - sig6)/dl_rij;
        
            for (int k = 0; k < 3; k++) {
                F[i*3 + k] += dif*rij[k];
                F[j*3 + k] -= dif*rij[k];
            }

            Epot[i*2] += sig12;
            Epot[i*2 + 1] -= sig6;
            Epot[j*2] += sig12;
            Epot[j*2 + 1] -= sig6;
        }

    post_up_cpu();
}

/*===========================================================================*
 *                  	aktualizacja sił wersje GPU		     	     *
 *===========================================================================*/

__global__ void up_forces_gpu_1(real_t* cords_gpu, real_t* F_gpu, real_t* Epot_gpu) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= N)
        return;

    for (int k = 0; k < 3; k++)
        F_gpu[i*3 + k] = 0;
    Epot_gpu[i*2] = 0;
    Epot_gpu[i*2 + 1] = 0;

    for (size_t j = 0; j < N; j++) {
        if (i == j)
            continue;

        real_t rij[3];
        for (int k = 0; k < 3; k++)
            rij[k] = cords_gpu[i*3 + k] - cords_gpu[j*3 + k];
        
        real_t dl_rij = 0;
        for (int k = 0; k < 3; k++)
            dl_rij += sq(rij[k]);

        if (rOd > 0.0 and dl_rij > sq(rOd))
            continue;

        real_t sig2 = sig*sig/dl_rij;
        real_t sig6 = sig2*sig2*sig2;
        real_t sig12 = sig6*sig6;

        real_t dif = (sig12 - sig6)/dl_rij;
        
        for (int k = 0; k < 3; k++)
            F_gpu[i*3 + k] += dif*rij[k];

        Epot_gpu[i*2] += sig12;
        Epot_gpu[i*2 + 1] -= sig6;
    }

    for (int k = 0; k < 3; k++)
        F_gpu[i*3 + k] *= 12.0*eps;
    Epot_gpu[i*2] *= eps/2.0;
    Epot_gpu[i*2 + 1] *= eps;
}

/* gpu wersja II */

/* zaokraglenie N w gore do pierwszej wielokrotnosci 32 */
constexpr int Nr32 = (N+31-(N+31)%32); 

__global__ void summer_gpu_2(real_t* F_gpu, real_t* Epot_gpu) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    for (int y = 1; y < Nr32/32; y++) {
        for (int k = 0; k < 3; k++)
            F_gpu[x*3 + k] += F_gpu[(x + y*Nr32)*3 + k];
        for (int k = 0; k < 2; k++)
            Epot_gpu[x*2 + k] += Epot_gpu[(x + y*Nr32)*2 + k];
    }

    for (int k = 0; k < 3; k++)
        F_gpu[x*3 + k] *= 12.0*eps;
    Epot_gpu[x*2] *= eps/2.0;
    Epot_gpu[x*2 + 1] *= eps;
}

__global__ void up_forces_gpu_2(real_t* cords, real_t* F_gpu, real_t* Epot_gpu) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.x + threadIdx.x;

    real_t C_host[3], C_guest[3];
    for (int k = 0; k < 3; k++) {
        C_host[k] = cords[x*3 + k];
        C_guest[k] = cords[y*3 + k];
    }

    int offset = x + blockIdx.y*Nr32;

    real_t F[3] = {0, 0, 0};
    real_t Epot[2] = {0, 0};

    for (int l = 0; l < 32; l++) {
        real_t rij[3];
        int j = blockIdx.y*32 + l;

        for (int k = 0; k < 3; k++)
            rij[k] = C_host[k] - __shfl_sync(static_cast<unsigned>(-1), C_guest[k], l);

        if (x == j or x >= N or j >= N)
            continue;
        
        real_t dl_rij = 0;
        for (int k = 0; k < 3; k++)
            dl_rij += sq(rij[k]);

        if (rOd > 0.0 and dl_rij > sq(rOd))
            continue;

        real_t sig2 = sig*sig/dl_rij;
        real_t sig6 = sig2*sig2*sig2;
        real_t sig12 = sig6*sig6;

        real_t dif = (sig12 - sig6)/dl_rij;
        
        for (int k = 0; k < 3; k++)
            F[k] += dif*rij[k];

        Epot[0] += sig12;
        Epot[1] -= sig6;
    }

    for (int k = 0; k < 3; k++)
        F_gpu[offset*3 + k] = F[k];
    Epot_gpu[offset*2] = Epot[0];
    Epot_gpu[offset*2 + 1] = Epot[1];
}

/* gpu III wersja, opt 3 zasada dynamiki Newtona */

__global__ void summer_gpu_3(real_t* F_gpu_host, real_t* Epot_gpu_host,
                             real_t* F_gpu_guest, real_t* Epot_gpu_guest) 
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    for (int y = 1; y < Nr32/32; y++) {
        for (int k = 0; k < 3; k++) {
            F_gpu_host[x*3 + k] += F_gpu_host[(x + y*Nr32)*3 + k];
            F_gpu_guest[x*3 + k] += F_gpu_guest[(x + y*Nr32)*3 + k];
        }
        for (int k = 0; k < 2; k++) {
            Epot_gpu_host[x*2 + k] += Epot_gpu_host[(x + y*Nr32)*2 + k];
            Epot_gpu_guest[x*2 + k] += Epot_gpu_guest[(x + y*Nr32)*2 + k];
        }
    }

    for (int k = 0; k < 3; k++) {
        F_gpu_host[x*3 + k] += F_gpu_guest[x*3 + k];
        F_gpu_host[x*3 + k] *= 12.0*eps;
    }

    for (int k = 0; k < 2; k++) {
        Epot_gpu_host[x*2 + k] += Epot_gpu_guest[x*2 + k];
        Epot_gpu_host[x*2 + k] *= eps;
    }

    Epot_gpu_host[x*2] /= 2.0;
}
__global__ void up_forces_gpu_3(real_t* cords, real_t* F_gpu_host, real_t* Epot_gpu_host,
                                real_t* F_gpu_guest, real_t* Epot_gpu_guest)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.x + threadIdx.x;
    int offset_host = x + blockIdx.y*Nr32;
    int offset_guest = y + blockIdx.x*Nr32;

    __shared__ real_t F_sh[32][3];
    __shared__ real_t Epot_sh[32][2];

    /* caly blok nie ma nic do liczenia */
    if (blockIdx.y*blockDim.x + 31 <= blockIdx.x*blockDim.x) {
        for (int k = 0; k < 3; k++) {
            F_gpu_host[offset_host*3 + k] = 0;
            F_gpu_guest[offset_guest*3 + k] = 0;
        }
        for (int k = 0; k < 2; k++) {
            Epot_gpu_host[offset_host*2 + k] = 0;
            Epot_gpu_guest[offset_guest*2 + k] = 0;
        }
        return;
    }

    for (int k = 0; k < 3; k++)
        F_sh[threadIdx.x][k] = 0;
    for (int k = 0; k < 2; k++)
        Epot_sh[threadIdx.x][k] = 0;

    real_t C_host[3], C_guest[3];
    for (int k = 0; k < 3; k++) {
        C_host[k] = cords[x*3 + k];
        C_guest[k] = cords[y*3 + k];
    }

    real_t F[3] = {0, 0, 0};
    real_t Epot[2] = {0, 0};

    __syncthreads();

    for (int poml = 0; poml < 32; poml++) {
        int l = (poml + threadIdx.x)%32;

        real_t rij[3];
        int j = blockIdx.y*32 + l;

        for (int k = 0; k < 3; k++)
            rij[k] = C_host[k] - __shfl_sync(static_cast<unsigned>(-1), C_guest[k], l);

        if (j <= x or x >= N or j >= N)
            continue;
        
        real_t dl_rij = 0;
        for (int k = 0; k < 3; k++)
            dl_rij += sq(rij[k]);

        if (rOd > 0.0 and dl_rij > sq(rOd))
            continue;

        real_t sig2 = sig*sig/dl_rij;
        real_t sig6 = sig2*sig2*sig2;
        real_t sig12 = sig6*sig6;

        real_t dif = (sig12 - sig6)/dl_rij;
        
        for (int k = 0; k < 3; k++) {
            F[k] += dif*rij[k];
            F_sh[l][k] -= dif*rij[k];
        }

        Epot[0] += sig12;
        Epot[1] -= sig6;
        Epot_sh[l][0] += sig12;
        Epot_sh[l][1] -= sig6;
    }

    __syncthreads();

    for (int k = 0; k < 3; k++) {
        F_gpu_host[offset_host*3 + k] = F[k];
        F_gpu_guest[offset_guest*3 + k] = F_sh[threadIdx.x][k];
    }
    for (int k = 0; k < 2; k++) {
        Epot_gpu_host[offset_host*2 + k] = Epot[k];
        Epot_gpu_guest[offset_guest*2 + k] = Epot_sh[threadIdx.x][k];
    }
}

/* gpu IV, listy sasiedztwa */
__global__ void rebuild_G_gpu(real_t* cords_gpu, int* G_gpu) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int pom = 0;
    
    if (x >= N)
        return;

    for (int j = 0; j < N; j++) {
        if (x == j)
            continue;
        real_t r = 0;
        for (int k = 0; k < 3; k++)
            r += sq(cords_gpu[x*3 + k]-cords_gpu[j*3 + k]);
        if (r <= sq(rOd + rBuff))
            G_gpu[x*MAX_LI + pom++] = j;
    }

    assert(pom < MAX_LI);
    G_gpu[x*MAX_LI + pom] = -1;
}

void check_G_gpu() {
    auto reb = [&](){
        rebuild_G_gpu<<<dim3(Nr32/32), dim3(32)>>>(cords_gpu, G_gpu);
        cudaAssert(cudaDeviceSynchronize());
        memcpy(poz0_cpu, cords, N*3*sizeof(real_t));
    };

    static bool first_builded = false;
    if (first_builded == false) {
        reb();
        first_builded = true;
        return;
    }

    for (int i = 0; i < N; i++) {
        real_t dl = 0;
        for (int k = 0; k < 3; k++)
            dl += sq(poz0_cpu[i*3 + k]-cords[i*3 + k]);
        if (dl*4.0 >= rBuff*rBuff) {
            reb();
            return;
        }
    }
}

__global__ void up_forces_gpu_4(real_t* cords_gpu, real_t* F_gpu, real_t* Epot_gpu, int* G_gpu) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    if (x >= N)
        return;

    real_t F[3] = {0, 0, 0};
    real_t Epot[2] = {0, 0};
    real_t C[3];

    for (int k = 0; k < 3; k++)
        C[k] = cords_gpu[x*3 + k];

    for (int pm = 0, j = G_gpu[x*MAX_LI + pm]; G_gpu[x*MAX_LI + pm] != -1; j = G_gpu[x*MAX_LI + ++pm]) {
        real_t rij[3];
        for (int k = 0; k < 3; k++)
            rij[k] = C[k] - cords_gpu[j*3 + k];
        
        real_t dl_rij = 0;
        for (int k = 0; k < 3; k++)
            dl_rij += sq(rij[k]);

        if (rOd > 0.0 and dl_rij > sq(rOd))
            continue;

        real_t sig2 = sig*sig/dl_rij;
        real_t sig6 = sig2*sig2*sig2;
        real_t sig12 = sig6*sig6;

        real_t dif = (sig12 - sig6)/dl_rij;
        
        for (int k = 0; k < 3; k++)
            F[k] += dif*rij[k];

        Epot[0] += sig12;
        Epot[1] -= sig6;
    }

    for (int k = 0; k < 3; k++)
        F_gpu[x*3 + k] = F[k]*12.0*eps;
    Epot_gpu[x*2] = Epot[0]*eps/2.0;
    Epot_gpu[x*2 + 1] = Epot[1]*eps;
}

/*===========================================================================*
 *                  	    uruchamianie symulacji		     	     *
 *===========================================================================*/

struct conf {
    string label;
    int co;
    dim3 blocks, threads;
};

void up_forces(const conf &config) {
    /* wersje cpu */
    switch (config.co) {
        case 0 :
            up_forces_cpu_1();
            return;
        case 1 :
            up_forces_cpu_2();
            return;
        case 2 :
            up_forces_cpu_3();
            return;
    }

    /* wersje gpu */
    cudaAssert(cudaMemcpy(cords_gpu, cords, N*3*sizeof(real_t), cudaMemcpyHostToDevice));

    switch (config.co) {
        case 3 : 
            up_forces_gpu_1<<<config.blocks, config.threads>>>(cords_gpu, F_gpu, Epot_gpu);
            cudaAssert(cudaMemcpy(F, F_gpu, N*3*sizeof(real_t), cudaMemcpyDeviceToHost));
            cudaAssert(cudaMemcpy(Epot, Epot_gpu, N*2*sizeof(real_t), cudaMemcpyDeviceToHost));
            break;
        case 4 :
            up_forces_gpu_2<<<config.blocks, config.threads>>>(cords_gpu, F_gpu, Epot_gpu);
            cudaAssert(cudaDeviceSynchronize());
            summer_gpu_2<<<dim3(Nr32/32), dim3(32)>>>(F_gpu, Epot_gpu);
            cudaAssert(cudaMemcpy(F, F_gpu, N*3*sizeof(real_t), cudaMemcpyDeviceToHost));
            cudaAssert(cudaMemcpy(Epot, Epot_gpu, N*2*sizeof(real_t), cudaMemcpyDeviceToHost));
            break;
        case 5 : 
            up_forces_gpu_3<<<config.blocks, config.threads>>>
                (cords_gpu, F_gpu_host, Epot_gpu_host, F_gpu_guest, Epot_gpu_guest);
            cudaAssert(cudaDeviceSynchronize());
            summer_gpu_3<<<dim3(Nr32/32), dim3(32)>>>(F_gpu_host, Epot_gpu_host, F_gpu_guest, Epot_gpu_guest);
            cudaAssert(cudaMemcpy(F, F_gpu_host, N*3*sizeof(real_t), cudaMemcpyDeviceToHost));
            cudaAssert(cudaMemcpy(Epot, Epot_gpu_host, N*2*sizeof(real_t), cudaMemcpyDeviceToHost));
            break;
        case 6 :
            check_G_gpu();
            up_forces_gpu_4<<<config.blocks, config.threads>>>(cords_gpu, F_gpu, Epot_gpu, G_gpu);
            cudaAssert(cudaDeviceSynchronize());
            cudaAssert(cudaMemcpy(F, F_gpu, N*3*sizeof(real_t), cudaMemcpyDeviceToHost));
            cudaAssert(cudaMemcpy(Epot, Epot_gpu, N*2*sizeof(real_t), cudaMemcpyDeviceToHost));
            break;
        default:
            assert(false and "Nieosiągalny switch");
    }

    cudaAssert(cudaDeviceSynchronize());
}

void sym_launcher(const conf &config) {
    log("Uruchamiam symulacje " + config.label, 1);
    log("Typ kernela = "+tos(config.co), 3);
    log("blocks = "+to_s(config.blocks.x, config.blocks.y), 3);
    log("threads = "+to_s(config.threads.x, config.threads.y), 3);

    start();
    
    switch (config.co) {
        case 3 :
            cudaAssert(cudaMalloc(&cords_gpu, N*3*sizeof(real_t)));
            cudaAssert(cudaMalloc(&F_gpu, N*3*sizeof(real_t)));
            cudaAssert(cudaMalloc(&Epot_gpu, N*2*sizeof(real_t)));
            break;
        case 4 :
            cudaAssert(cudaMalloc(&cords_gpu, Nr32*3*sizeof(real_t)));
            cudaAssert(cudaMalloc(&F_gpu, 3*Nr32*Nr32/32*sizeof(real_t)));
            cudaAssert(cudaMalloc(&Epot_gpu, 2*Nr32*Nr32/32*sizeof(real_t)));
            break;
        case 5 :
            cudaAssert(cudaMalloc(&cords_gpu, Nr32*3*sizeof(real_t))); 
            cudaAssert(cudaMalloc(&F_gpu_host, 3*Nr32*Nr32/32*sizeof(real_t)));
            cudaAssert(cudaMalloc(&F_gpu_guest, 3*Nr32*Nr32/32*sizeof(real_t)));
            cudaAssert(cudaMalloc(&Epot_gpu_host, 2*Nr32*Nr32/32*sizeof(real_t)));
            cudaAssert(cudaMalloc(&Epot_gpu_guest, 2*Nr32*Nr32/32*sizeof(real_t)));
            break;
        case 6 :
            cudaAssert(cudaMalloc(&cords_gpu, N*3*sizeof(real_t)));
            cudaAssert(cudaMalloc(&F_gpu, N*3*sizeof(real_t)));
            cudaAssert(cudaMalloc(&Epot_gpu, N*2*sizeof(real_t)));
            cudaAssert(cudaMalloc(&G_gpu, N*MAX_LI*sizeof(int)));
            break;
    }
    
    init_atomow();

    for (int nr = 0; nr < Nterm+Ngrz+Nch; nr++) {
        up_V();
        up_cords();

        up_forces(config);
        up_ballon();

        up_V();
        
        up_kin();
        up_temp(nr);

        up_logi(nr);
        up_stats(nr);
    }
    up_logi(0);
    stats_out();
    
    switch (config.co) {
        case 3 :
            cudaAssert(cudaFree(cords_gpu));
            cudaAssert(cudaFree(F_gpu));
            cudaAssert(cudaFree(Epot_gpu));
            break;
        case 4 :
            cudaAssert(cudaFree(cords_gpu));
            cudaAssert(cudaFree(F_gpu));
            cudaAssert(cudaFree(Epot_gpu));
            break;
        case 5 :
            cudaAssert(cudaFree(cords_gpu));
            cudaAssert(cudaFree(F_gpu_host));
            cudaAssert(cudaFree(F_gpu_guest));
            cudaAssert(cudaFree(Epot_gpu_host));
            cudaAssert(cudaFree(Epot_gpu_guest));
            break;
        case 6 :
            cudaAssert(cudaFree(cords_gpu));
            cudaAssert(cudaFree(F_gpu));
            cudaAssert(cudaFree(Epot_gpu));
            cudaAssert(cudaFree(G_gpu));
            break;
    }

    stop();
    log("Koniec symulacji, czas trwania = " + tos(get_duration()), 1);
}

int main() {
    vector<conf> v_conf({
        {"cpu_1", 0, {1}, {1}},
        {"cpu_2", 1, {1}, {1}},
        {"cpu_3", 2, {1}, {1}},
        {"gpu_1", 3, {Nr32/32}, {32}},
        {"gpu_2", 4, {Nr32/32, Nr32/32}, {32}},
        {"gpu_3", 5, {Nr32/32, Nr32/32}, {32}},
        {"gpu_4", 6, {Nr32/32}, {32}},
    });

    for (auto cf : v_conf) 
        sym_launcher(cf);
}
