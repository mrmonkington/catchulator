from django.shortcuts import render
from catchsim import sim_run

# Create your views here.

def start(request):
    return render(request, 'ui/start.html')

def catchulate(request):
    c = {}
    postcode = request.POST.get('postcode')
    fsm = request.POST.get('fsm', '') != ''
    option = request.POST.get('option')
    prefs=[]
    prefs.append(request.POST.get('prefs1'))
    prefs.append(request.POST.get('prefs2'))
    prefs.append(request.POST.get('prefs3'))
    print(prefs)
    panyear = request.POST.get('panyear')
    popyear = request.POST.get('popyear')
    c["disp"] = sim_run(postcode, option, popyear, panyear, prefs, False, None, None, None, fsm)
    return render(request, 'ui/catchulate.html', context=c)



